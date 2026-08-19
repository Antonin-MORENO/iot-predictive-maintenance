"""
Real-time IoT sensor simulator.

Replays the NASA C-MAPSS dataset (turbofan engine degradation) row by
row, engine by engine, with a delay between each cycle to simulate a
live sensor stream.

Three output modes:
- "print" : prints readings to the terminal (no broker required, useful
            to validate the logic before connecting to the cloud)
- "mqtt"  : publishes each reading to an MQTT broker (local, e.g.
            Mosquitto)
- "azure" : publishes each reading to Azure IoT Hub, using the device
            connection string from a local .env file

Expected format of the source file (train_FD001.txt, space-separated,
no header): see README.md for the download link.
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

COLUMN_NAMES = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_cmapss(filepath: Path) -> pd.DataFrame:
    """Load a train_FDxxx.txt file from the C-MAPSS dataset."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def make_reading(row: pd.Series) -> dict:
    """Build the sensor message sent for a given cycle."""
    return {
        "unit_id": int(row["unit_id"]),
        "cycle": int(row["cycle"]),
        "timestamp": time.time(),
        "sensors": {
            col: float(row[col])
            for col in row.index
            if col.startswith("sensor_")
        },
    }


def stream_print(df: pd.DataFrame, unit_id: int, delay: float) -> None:
    unit_df = df[df["unit_id"] == unit_id].sort_values("cycle")
    for _, row in unit_df.iterrows():
        reading = make_reading(row)
        print(json.dumps(reading))
        time.sleep(delay)


def stream_mqtt(
    df: pd.DataFrame,
    unit_id: int,
    delay: float,
    broker_host: str,
    broker_port: int,
    topic: str,
) -> None:
    import paho.mqtt.client as mqtt

    client = mqtt.Client()
    client.connect(broker_host, broker_port)
    client.loop_start()

    unit_df = df[df["unit_id"] == unit_id].sort_values("cycle")
    for _, row in unit_df.iterrows():
        reading = make_reading(row)
        client.publish(topic, json.dumps(reading))
        time.sleep(delay)

    client.loop_stop()
    client.disconnect()


def stream_azure(df: pd.DataFrame, unit_id: int, delay: float) -> None:
    """Publish the stream to Azure IoT Hub, using the device registered in .env.

    Reuses the exact same reading/message-building logic as
    stream_print() and stream_mqtt() — only the way the message is sent
    changes.
    """
    from azure.iot.device import IoTHubDeviceClient, Message

    load_dotenv()
    connection_string = os.getenv("AZURE_IOT_CONNECTION_STRING")
    if not connection_string:
        raise RuntimeError(
            "AZURE_IOT_CONNECTION_STRING is missing. "
            "Create a .env file at the project root with this variable "
            "(see .env.example)."
        )

    client = IoTHubDeviceClient.create_from_connection_string(connection_string)
    client.connect()
    print("Connected to Azure IoT Hub.")

    unit_df = df[df["unit_id"] == unit_id].sort_values("cycle")
    for _, row in unit_df.iterrows():
        reading = make_reading(row)
        message = Message(json.dumps(reading))
        client.send_message(message)
        print(f"Sent: cycle {reading['cycle']}")
        time.sleep(delay)

    client.disconnect()
    print("Disconnected.")


def main() -> None:
    parser = argparse.ArgumentParser(description="IoT sensor simulator")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "train_FD001.txt",
        help="Path to the train_FD001.txt file",
    )
    parser.add_argument("--unit-id", type=int, default=1, help="Engine ID to replay")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between cycles (seconds)")
    parser.add_argument("--mode", choices=["print", "mqtt", "azure"], default="print")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--topic", default="sensors/turbofan")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"File not found: {args.data}\n"
            "Download train_FD001.txt (see README.md) and place it in data/."
        )

    df = load_cmapss(args.data)

    if args.mode == "print":
        stream_print(df, args.unit_id, args.delay)
    elif args.mode == "mqtt":
        stream_mqtt(df, args.unit_id, args.delay, args.broker_host, args.broker_port, args.topic)
    elif args.mode == "azure":
        stream_azure(df, args.unit_id, args.delay)


if __name__ == "__main__":
    main()