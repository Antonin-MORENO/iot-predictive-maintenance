"""
IoT Hub consumer: reads sensor messages from the IoT Hub built-in
Event Hub-compatible endpoint, scores each one via the Azure Function
endpoint, and appends the result to a local JSONL file that the
Streamlit dashboard reads.

This is the missing link that connects the ingestion flow (simulator ->
IoT Hub) to the scoring flow (model endpoint), forming the full
real-time pipeline.

Environment variables (in .env):
  IOTHUB_EVENTHUB_CONNECTION_STRING  -> read the message stream
  SCORING_ENDPOINT_URL               -> full Azure Function URL with ?code=
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from azure.eventhub import EventHubConsumerClient
from dotenv import load_dotenv

load_dotenv()

CONNECTION_STRING = os.getenv("IOTHUB_EVENTHUB_CONNECTION_STRING")
SCORING_URL = os.getenv("SCORING_ENDPOINT_URL")
OUTPUT_FILE = Path(__file__).resolve().parents[2] / "data" / "scored_stream.jsonl"

CONSUMER_GROUP = "$Default"


def score_reading(sensors: dict) -> dict | None:
    """Call the Azure Function scoring endpoint for one reading."""
    try:
        resp = requests.post(SCORING_URL, json={"sensors": sensors}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"Scoring request failed: {e}")
        return None


def on_event(partition_context, event):
    if event is None:
        return
    try:
        body = json.loads(event.body_as_str())
    except (ValueError, TypeError):
        print("Skipping non-JSON message.")
        partition_context.update_checkpoint(event)
        return

    sensors = body.get("sensors")
    if not sensors:
        partition_context.update_checkpoint(event)
        return

    result = score_reading(sensors)
    if result is not None:
        record = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "unit_id": body.get("unit_id"),
            "cycle": body.get("cycle"),
            "reconstruction_error": result["reconstruction_error"],
            "threshold": result["threshold"],
            "is_anomaly": result["is_anomaly"],
        }
        OUTPUT_FILE.parent.mkdir(exist_ok=True)
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        flag = "ANOMALY" if record["is_anomaly"] else "ok"
        print(f"cycle {record['cycle']}: error={record['reconstruction_error']:.3f} [{flag}]")

    partition_context.update_checkpoint(event)


def main():
    if not CONNECTION_STRING or not SCORING_URL:
        raise RuntimeError(
            "Missing env vars. Set IOTHUB_EVENTHUB_CONNECTION_STRING and "
            "SCORING_ENDPOINT_URL in your .env file (see .env.example)."
        )

    client = EventHubConsumerClient.from_connection_string(
        CONNECTION_STRING, consumer_group=CONSUMER_GROUP
    )
    print("Listening to IoT Hub stream... (Ctrl+C to stop)")
    try:
        with client:
            # starting_position="-1" reads only new messages from now on
            client.receive(on_event=on_event, starting_position="-1")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
