#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP="kafka:29092"

topics=(
  "content.raw"
  "content.classified"
)

for t in "${topics[@]}"; do
  docker compose -f infra/docker-compose.yml exec -T kafka \
    kafka-topics --bootstrap-server "$BOOTSTRAP" \
    --create --if-not-exists --topic "$t" --partitions 3 --replication-factor 1
done

docker compose -f infra/docker-compose.yml exec -T kafka \
  kafka-topics --bootstrap-server "$BOOTSTRAP" --list
