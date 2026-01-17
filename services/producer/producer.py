import json
import time
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer


BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "content.raw"


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_event(content_id: str, text: str) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "created_at": now_iso_utc(),
        "content_id": content_id,
        "text": text,
    }


def main():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        acks="all",
        linger_ms=10,
    )

    # Mix: neutral + toxic-ish examples (for testing model)
    messages = [
        ("c_100", "Thanks for the update, great work!"),
        ("c_101", "Please fix the bug, I can reproduce it on Windows."),
        ("c_102", "This feature is confusing, could you explain it?"),
        ("c_103", "You are stupid and useless."),
        ("c_104", "Shut up, nobody cares about your opinion."),
        ("c_105", "This is nonsense. Stop talking."),
        ("c_106", "I hate you."),
        ("c_107", "Go away."),
        ("c_108", "You're doing a good job, keep it up!"),
        ("c_109", "What a terrible idea, but maybe it can be improved."),
        ("c_110", "I will report this to the admin."),
        ("c_111", "You are an idiot."),
        ("c_112", "Let's discuss this calmly and find a solution."),
        ("c_113", "This is the worst thing I've ever seen."),
        ("c_114", "Could you share the steps to reproduce the issue?"),
    ]

    print(f"Sending {len(messages)} events to Kafka topic={TOPIC} ...\n")

    for content_id, text in messages:
        event = make_event(content_id, text)
        producer.send(TOPIC, key=content_id, value=event)
        print(f"Sent key={content_id} event_id={event['event_id']}")
        print(json.dumps(event, ensure_ascii=False, indent=2))
        print("-" * 60)
        time.sleep(0.3)  # small delay so you can see streaming live

    producer.flush()
    producer.close()
    print("\nDONE ✅")


if __name__ == "__main__":
    main()
