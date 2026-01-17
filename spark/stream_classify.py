from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType


def main():
    spark = (
        SparkSession.builder
        .appName("moderation-stream")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Schema of incoming JSON (Kafka value)
    schema = StructType([
        StructField("event_id", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("content_id", StringType(), True),
        StructField("text", StringType(), True),
    ])

    # 1) Read stream from Kafka topic: content.raw
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "content.raw")
        .option("startingOffsets", "earliest")
        .load()
    )

    # 2) Parse Kafka value as JSON -> columns
    parsed = (
        raw.select(
            F.col("key").cast("string").alias("key"),
            F.col("value").cast("string").alias("value"),
            F.col("timestamp").alias("kafka_timestamp"),
        )
        .withColumn("json", F.from_json(F.col("value"), schema))
        .select(
            F.col("key"),
            F.col("json.event_id").alias("event_id"),
            F.col("json.created_at").alias("created_at"),
            F.col("json.content_id").alias("content_id"),
            F.col("json.text").alias("text"),
            F.col("kafka_timestamp")
        )
    )

    # 3) Simple rule-based classification (temporary)
    text_lc = F.lower(F.col("text"))
    is_bad = (
        text_lc.contains("free money") |
        text_lc.contains("click here") |
        text_lc.contains("http") |
        text_lc.contains("idiot") |
        text_lc.contains("stupid")
    )

    # Add prediction + timing info
    out = (
        parsed
        .withColumn("prediction", F.when(is_bad, F.lit(1)).otherwise(F.lit(0)))
        .withColumn("model_version", F.lit("rules_v0"))
        .withColumn("processed_at", F.current_timestamp())
        .withColumn("created_ts", F.to_timestamp("created_at"))
        .withColumn(
            "latency_ms",
            (F.col("processed_at").cast("long") - F.col("created_ts").cast("long")) * 1000
        )
        .withColumn(
            "out_json",
            F.to_json(F.struct(
                "event_id", "created_at", "content_id", "text",
                "prediction", "model_version", "processed_at", "latency_ms"
            ))
        )
        .select(
            F.col("content_id").alias("out_key"),
            F.col("out_json").alias("out_value")
        )
    )

    # 4) Write to Kafka topic: content.classified
    query = (
        out.select(
            F.col("out_key").cast("string").alias("key"),
            F.col("out_value").cast("string").alias("value"),
        )
        .writeStream.format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("topic", "content.classified")
        .option("checkpointLocation", "jobs/checkpoints/moderation_stream_v0")
        .outputMode("append")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
