"""
Simulateur de capteurs IoT temps réel.

Rejoue le dataset NASA C-MAPSS (dégradation de turboréacteurs) ligne par
ligne, moteur par moteur, en respectant un délai entre chaque cycle pour
simuler un flux de capteurs en direct.

Deux modes de sortie :
- "print"  : affiche les mesures dans le terminal (aucun broker requis,
             utile pour valider la logique avant de brancher le cloud)
- "mqtt"   : publie chaque mesure sur un broker MQTT (local ou IoT Hub /
             IoT Core une fois la connexion cloud configurée)

Format attendu du fichier source (train_FD001.txt, séparateur espace,
sans en-tête) : voir README.md pour le lien de téléchargement.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

COLUMN_NAMES = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_cmapss(filepath: Path) -> pd.DataFrame:
    """Charge un fichier train_FDxxx.txt du dataset C-MAPSS."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def make_reading(row: pd.Series) -> dict:
    """Construit le message capteur envoyé pour un cycle donné."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulateur de capteurs IoT")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "train_FD001.txt",
        help="Chemin vers le fichier train_FD001.txt",
    )
    parser.add_argument("--unit-id", type=int, default=1, help="ID du moteur à rejouer")
    parser.add_argument("--delay", type=float, default=0.5, help="Délai entre deux cycles (secondes)")
    parser.add_argument("--mode", choices=["print", "mqtt"], default="print")
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--topic", default="sensors/turbofan")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {args.data}\n"
            "Télécharge train_FD001.txt (voir README.md) et place-le dans data/."
        )

    df = load_cmapss(args.data)

    if args.mode == "print":
        stream_print(df, args.unit_id, args.delay)
    else:
        stream_mqtt(df, args.unit_id, args.delay, args.broker_host, args.broker_port, args.topic)


if __name__ == "__main__":
    main()
