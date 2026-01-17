import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def add_binary_label(df):
    # label = 1 if ANY of the toxicity flags is 1 else 0
    cond = None
    for c in LABEL_COLS:
        c1 = (F.col(c) == 1)
        cond = c1 if cond is None else (cond | c1)
    return df.withColumn("label", F.when(cond, F.lit(1.0)).otherwise(F.lit(0.0)))


def confusion_metrics(pred_df):
    """
    Confusion matrix + P/R/F1/ACC computed with pure Spark SQL counts
    (more stable than RDD-based MulticlassMetrics on Windows).
    """
    p = pred_df.select(
        F.col("prediction").cast("int").alias("p"),
        F.col("label").cast("int").alias("y")
    )

    tn = p.filter((F.col("p") == 0) & (F.col("y") == 0)).count()
    fp = p.filter((F.col("p") == 1) & (F.col("y") == 0)).count()
    fn = p.filter((F.col("p") == 0) & (F.col("y") == 1)).count()
    tp = p.filter((F.col("p") == 1) & (F.col("y") == 1)).count()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return tn, fp, fn, tp, float(precision), float(recall), float(f1), float(acc)


def build_pipeline(model):
    # Simple and fast NLP feature engineering (good baseline for report)
    tokenizer = RegexTokenizer(inputCol="comment_text", outputCol="tokens", pattern="\\W+")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    cv = CountVectorizer(inputCol="filtered", outputCol="tf", vocabSize=100_000, minDF=5)
    idf = IDF(inputCol="tf", outputCol="features")
    return Pipeline(stages=[tokenizer, remover, cv, idf, model])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to train.csv")
    parser.add_argument("--output", required=True, help="Where to save best Spark PipelineModel")
    parser.add_argument("--sample", type=float, default=1.0, help="Use fraction of data (e.g. 0.2 for faster)")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("train-binary-moderation")
        # fewer workers -> fewer Windows networking issues
        .master("local[2]")
        # important on Windows: stable Python worker <-> JVM networking
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.network.timeout", "600s")
        .config("spark.python.worker.reuse", "true")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Read CSV robustly (handles quotes/escapes)
    df = (
        spark.read
        .option("header", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(args.input)
    )

    # Keep only needed columns
    df = df.select("comment_text", *LABEL_COLS).na.drop(subset=["comment_text"])

    # Safe cast label columns (tolerate malformed -> null)
    for c in LABEL_COLS:
        df = df.withColumn(c, F.expr(f"try_cast({c} as int)"))

    # Drop rows where any label failed to parse
    df = df.na.drop(subset=LABEL_COLS)

    # Create binary label
    df = add_binary_label(df).select("comment_text", "label")

    # Optional sampling for speed
    if args.sample < 1.0:
        df = df.sample(False, args.sample, seed=42)

    # Class balance (useful for report)
    print("\nClass balance (label -> count):")
    for row in df.groupBy("label").count().orderBy("label").collect():
        print(row)

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    models = {
        "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=50),
        "RandomForest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=80, maxDepth=12),
        "GBT": GBTClassifier(featuresCol="features", labelCol="label", maxIter=40, maxDepth=5),
    }

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    results = []
    fitted = {}

    for name, clf in models.items():
        print(f"\nTraining: {name}")
        pipe = build_pipeline(clf)
        m = pipe.fit(train)
        pred = m.transform(test)

        auc = evaluator_auc.evaluate(pred)
        tn, fp, fn, tp, precision, recall, f1, acc = confusion_metrics(pred)

        results.append((name, auc, acc, precision, recall, f1, tn, fp, fn, tp))
        fitted[name] = m

        print(f"{name} AUC={auc:.4f} ACC={acc:.4f} P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
        print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

    results.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Comparison (sorted by AUC) ===")
    print("Model | AUC | ACC | Precision | Recall | F1 | TN FP FN TP")
    for r in results:
        print(f"{r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]} {r[7]} {r[8]} {r[9]}")

    best_name = results[0][0]
    fitted[best_name].write().overwrite().save(args.output)
    print(f"\nSaved BEST model: {best_name} -> {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()
