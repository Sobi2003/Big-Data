# Streaming Moderation System (Kafka + Spark + Docker)

## About
End-to-end **real-time pipeline** for text moderation. A Python producer publishes events to Kafka, **Spark Structured Streaming** classifies them (rules or ML), and a **FastAPI** app displays the latest results from the output topic.

## Tech stack
Kafka (Docker Compose), Kafka UI, Spark / PySpark (Structured Streaming, MLlib), Python (FastAPI, kafka-python)

## Architecture
Producer → `content.raw` → Spark Streaming → `content.classified` → FastAPI UI

## Quickstart

### 1) Start Kafka + Kafka UI
docker compose -f infra/docker-compose.yml up -d  
Kafka UI: http://localhost:8080

### 2) Create topics
bash scripts/create-topics.sh

### 3) Run the Spark streaming job (rule-based)
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 spark/stream_classify.py

### 4) Send sample events
python -m pip install -r services/producer/requirements.txt  
python services/producer/producer.py

### 5) Run the consumer UI
python -m pip install -r services/consumer_api/requirements.txt  
uvicorn services.consumer_api.app:app --reload --port 8000

Open:
- UI: http://localhost:8000
- JSON: http://localhost:8000/latest

## Repo structure
- `infra/` – Docker Compose (Kafka, Zookeeper, Kafka UI)
- `scripts/` – topic creation helpers
- `spark/` – streaming jobs + training scripts
- `services/producer/` – Python producer
- `services/consumer_api/` – FastAPI UI/API
- `data/` – sample datasets
