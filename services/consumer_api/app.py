import json
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from kafka import KafkaConsumer, TopicPartition

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "content.classified"

app = FastAPI(title="Moderation Consumer API", version="1.1")


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {"_raw": s, "_parse_error": True}


def fetch_last_messages(n: int = 20, timeout_ms: int = 1500) -> List[Dict[str, Any]]:
    consumer = KafkaConsumer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        enable_auto_commit=False,
        auto_offset_reset="latest",
        consumer_timeout_ms=timeout_ms,
        value_deserializer=lambda v: v.decode("utf-8", errors="replace"),
        key_deserializer=lambda v: v.decode("utf-8", errors="replace") if v else None,
    )

    try:
        partitions = consumer.partitions_for_topic(TOPIC)
        if not partitions:
            return []

        tps = [TopicPartition(TOPIC, p) for p in sorted(partitions)]
        consumer.assign(tps)

        consumer.seek_to_end(*tps)
        end_offsets = consumer.end_offsets(tps)

        for tp in tps:
            end = end_offsets.get(tp, 0)
            start = max(0, end - n)
            consumer.seek(tp, start)

        rows: List[Dict[str, Any]] = []
        for msg in consumer:
            payload = _safe_json_loads(msg.value)
            rows.append(
                {
                    "partition": msg.partition,
                    "offset": msg.offset,
                    "key": msg.key,
                    "kafka_timestamp": datetime.fromtimestamp(msg.timestamp / 1000.0).isoformat(),
                    "value": payload,
                }
            )

        rows.sort(key=lambda r: (r["kafka_timestamp"], r["partition"], r["offset"]))
        return rows[-n:]
    finally:
        consumer.close()


def render_html(rows: List[Dict[str, Any]], n: int) -> str:
    def cell(x: Any) -> str:
        if x is None:
            return ""
        s = str(x)
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    trs = []
    for r in rows:
        v = r.get("value", {}) or {}
        text = v.get("text", "")
        pred = v.get("prediction", "")
        decision = v.get("decision", "")  # <-- NOWE
        prob = v.get("prob_toxic", "")
        model = v.get("model_version", "")
        processed = v.get("processed_at", "")
        latency = v.get("latency_ms", "")

        trs.append(
            f"""
            <tr>
              <td>{cell(r.get("partition"))}</td>
              <td>{cell(r.get("offset"))}</td>
              <td>{cell(r.get("key"))}</td>
              <td style="max-width:520px; white-space:normal;">{cell(text)}</td>
              <td>{cell(pred)}</td>
              <td>{cell(decision)}</td>
              <td>{cell(prob)}</td>
              <td>{cell(model)}</td>
              <td>{cell(processed)}</td>
              <td>{cell(latency)}</td>
            </tr>
            """
        )

    table_rows = "\n".join(trs) if trs else "<tr><td colspan='10'>Brak danych</td></tr>"

    return f"""
<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Moderation – Consumer UI</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; }}
    .top {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
    button {{ padding: 8px 12px; cursor:pointer; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; text-align: left; }}
    .hint {{ color: #666; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="top">
    <h2 style="margin:0;">Moderation Consumer UI</h2>
    <form method="get" action="/" style="margin:0;">
      <label>Ostatnie N:
        <input type="number" name="n" value="{n}" min="1" max="500" style="width:90px;"/>
      </label>
      <button type="submit">Odśwież</button>
    </form>
    <a href="/latest?n={n}">JSON</a>
  </div>

  <div class="hint">
    Źródło: Kafka topic <b>{TOPIC}</b> (bootstrap: {BOOTSTRAP_SERVERS})
  </div>

  <table>
    <thead>
      <tr>
        <th>Partition</th>
        <th>Offset</th>
        <th>Key</th>
        <th>Text</th>
        <th>Prediction</th>
        <th>Decision</th>
        <th>Prob toxic</th>
        <th>Model</th>
        <th>Processed at</th>
        <th>Latency ms</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def ui(n: int = Query(20, ge=1, le=500)):
    rows = fetch_last_messages(n=n)
    return HTMLResponse(render_html(rows, n=n))


@app.get("/latest")
def latest(n: int = Query(20, ge=1, le=500)):
    rows = fetch_last_messages(n=n)
    return JSONResponse({"topic": TOPIC, "count": len(rows), "items": rows})
