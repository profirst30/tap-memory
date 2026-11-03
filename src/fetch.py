"""Download SKAB (Skoltech Anomaly Benchmark, GPL-3.0) and read its experiments.

    python -m src.fetch

The archive is extracted to RAW, outside the repository.
"""
import csv
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("TAP_RAW", Path.home() / "skab_raw"))
URL = "https://github.com/waico/SKAB/archive/refs/heads/master.zip"
DATA = RAW / "SKAB-master" / "data"
SENSORS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature", "Thermocouple", "Voltage",
           "Volume Flow RateRMS"]


def download() -> None:
    if DATA.exists():
        return
    RAW.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=300) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(RAW)


def read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Sensor matrix (n, 8) and anomaly labels (n,), one row per second."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    head, body = rows[0], rows[1:]
    x = np.array([[float(r[head.index(s)]) for s in SENSORS] for r in body], np.float32)
    a = np.array([int(float(r[head.index("anomaly")])) for r in body], np.int8) if "anomaly" in head else np.zeros(len(body), np.int8)
    return x, a


def experiments() -> list[Path]:
    return sorted(p for d in ("valve1", "valve2", "other") for p in (DATA / d).glob("*.csv"))


def normal() -> Path:
    return DATA / "anomaly-free" / "anomaly-free.csv"


if __name__ == "__main__":
    download()
    print(len(experiments()), "expériences ;", normal())
