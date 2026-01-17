import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def add_binary_label(df):
    cond = None
    for c in LABEL_COLS:
        c1 = (F.col(c) == 1)
        cond = c1 if cond is None else (cond | c1)
    return df.withColumn("label", F.when(cond, F.lit(1.0)).otherwise(F.lit(0.0)))


def confusion_counts(pred_df):
    p = pred_df.select(F.col("prediction").cast("int").alias("p"), F.col("label").cast("int").alias("y"))
    tn = p.filter((F.col("p") == 0) & (F.col("y") == 0)).count()
    fp = p.filter((F.col("p") == 1) & (F.col("y") == 0)).count()
    fn = p.filter((F.col("p") == 0) & (F.col("y") == 1)).count()
    tp = p.filter((F.col("p") == 1) & (F.col("y") == 1)).count()
    return tn, fp, fn, tp


def prf1_from_counts(tn, fp, fn, tp):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return float(precision), float(recall), float(f1), float(acc)


def build_pipeline(gbt, vocab_size=100_000, min_df=5):
    tokenizer = RegexTokenizer(inputCol="comment_text", outputCol="tokens", pattern="\\W+")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    cv = CountVectorizer(inputCol="filtered", outputCol="tf", vocabSize=vocab_size, minDF=min_df)
    idf = IDF(inputCol="tf", outputCol="features")
    return Pipeline(stages=[tokenizer, remover, cv, idf, gbt])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=float, default=0.2)
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("tune-gbt-moderation")
        .master("local[2]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.network.timeout", "600s")
        .config("spark.python.worker.reuse", "true")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read
        .option("header", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(args.input)
    )

    df = df.select("comment_text", *LABEL_COLS).na.drop(subset=["comment_text"])
    for c in LABEL_COLS:
        df = df.withColumn(c, F.expr(f"try_cast({c} as int)"))
    df = df.na.drop(subset=LABEL_COLS)

    df = add_binary_label(df).select("comment_text", "label")
    if args.sample < 1.0:
        df = df.sample(False, args.sample, seed=42)

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    gbt = GBTClassifier(featuresCol="features", labelCol="label")
    pipe = build_pipeline(gbt)

    # Param grid (Grid Search)
    paramGrid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [3, 5])
        .addGrid(gbt.maxIter, [20, 40])
        .addGrid(gbt.stepSize, [0.05, 0.1])
        .build()
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )

    tvs = TrainValidationSplit(
        estimator=pipe,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=1,  # stabilniej na Windows
        seed=42
    )

    tvsModel = tvs.fit(train)
    bestModel = tvsModel.bestModel

    pred = bestModel.transform(test)
    auc = evaluator.evaluate(pred)
    tn, fp, fn, tp = confusion_counts(pred)
    precision, recall, f1, acc = prf1_from_counts(tn, fp, fn, tp)

    print("\n=== BEST GBT after tuning ===")
    # Odczyt parametrów z ostatniego stage (GBTClassifierModel)
    gbt_model = bestModel.stages[-1]
    print(f"Params: maxDepth={gbt_model.getMaxDepth()}, maxIter={gbt_model.getMaxIter()}, stepSize={gbt_model.getStepSize()}")
    print(f"AUC={auc:.4f} ACC={acc:.4f} P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

    bestModel.write().overwrite().save(args.output)
    print(f"\nSaved tuned model -> {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()
