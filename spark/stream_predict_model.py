from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

MODEL_PATH = "models/moderation_pipeline_gbt_tuned"  # <-- Twój tuned model
THRESHOLD = 0.30  # próg decyzji dla klasy toxic (1)


def main():
    spark = (
        SparkSession.builder
        .appName("moderation-stream-ml")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    schema = StructType([
        StructField("event_id", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("content_id", StringType(), True),
        StructField("text", StringType(), True),
    ])

    # 1) Read from Kafka (content.raw)
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "content.raw")
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        raw.select(
            F.col("key").cast("string").alias("key"),
            F.col("value").cast("string").alias("value"),
            F.col("timestamp").alias("kafka_timestamp"),
        )
        .withColumn("json", F.from_json(F.col("value"), schema))
        .select(
            F.col("json.event_id").alias("event_id"),
            F.col("json.created_at").alias("created_at"),
            F.col("json.content_id").alias("content_id"),
            F.col("json.text").alias("comment_text"),
            F.col("kafka_timestamp"),
        )
        .filter(F.col("event_id").isNotNull() & F.col("comment_text").isNotNull())
    )

    # 2) Load trained PipelineModel and predict
    model = PipelineModel.load(MODEL_PATH)
    scored = model.transform(parsed)

    # probability is VectorUDT -> convert to array, take P(class=1)
    scored = scored.withColumn("prob_toxic", vector_to_array(F.col("probability"))[1])

    # our custom decision threshold
    scored = scored.withColumn(
        "decision",
        F.when(F.col("prob_toxic") >= F.lit(THRESHOLD), F.lit(1)).otherwise(F.lit(0))
    )

    # 3) Timing / latency in true milliseconds
    scored = scored.withColumn("created_ts", F.to_timestamp("created_at"))
    scored = scored.withColumn("processed_at", F.current_timestamp())
    scored = scored.withColumn("latency_ms", (F.unix_millis("processed_at") - F.unix_millis("created_ts")))

    # 4) Output JSON -> Kafka (content.classified)
    out = (
        scored
        .withColumn("model_version", F.lit("gbt_tuned_v1"))
        .withColumn(
            "out_json",
            F.to_json(F.struct(
                "event_id", "created_at", "content_id",
                F.col("comment_text").alias("text"),
                F.col("prediction").cast("int").alias("prediction"),  # Spark default threshold 0.5
                F.col("decision").cast("int").alias("decision"),      # Our THRESHOLD
                F.col("prob_toxic").alias("prob_toxic"),
                "model_version", "processed_at", "latency_ms"
            ))
        )
        .select(
            F.col("content_id").alias("out_key"),
            F.col("out_json").alias("out_value")
        )
    )

    query = (
        out.select(
            F.col("out_key").cast("string").alias("key"),
            F.col("out_value").cast("string").alias("value")
        )
        .writeStream.format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("topic", "content.classified")
        .option("checkpointLocation", "jobs/checkpoints/moderation_stream_mllib_v2")
        .outputMode("append")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
