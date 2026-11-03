"""What the microcontroller computes between two phone taps, and what the phone can conclude from what it reads.

Units: one row per second; the MCU works on 10-s windows. Everything is causal.
"""
import numpy as np

CAL = 300  # seconds of normal running after installation, used to learn the machine's own normal levels
W = 10  # seconds per window
N_SENSORS = 8
SAMPLE_BYTES = 2 * N_SENSORS  # one raw sample: 8 int16 values
STATE_BYTES = 2 * 2 * N_SENSORS  # last window state: mean and spread of each sensor, int16
HEADER_BYTES = 8  # summary header: windows flagged (u16), first and last flagged window (u16 each), top sensor (u8), version (u8)
Q_THRESHOLD = 64  # in the summary, a score equal to the decision threshold is stored as 64 (uint8)


def standardize(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Scale each sensor by its mean and spread during calibration. The spread is floored so that a sensor that did
    not move during calibration (a constant flow rate, for instance) does not turn every small change into an alarm."""
    m, s = ref.mean(0), ref.std(0)
    s = np.maximum(s, np.maximum(0.02 * np.abs(m), 1e-3))
    return ((x - m) / s).astype(np.float32)


def features(z: np.ndarray, w: int = W) -> np.ndarray:
    """Mean and spread of each sensor over consecutive windows of w samples (a shorter last window is dropped,
    except when there is less than one full window: then the whole input is one window)."""
    if len(z) < w:
        return np.concatenate([z.mean(0), z.std(0)])[None].astype(np.float32)
    n = len(z) // w
    b = z[:n * w].reshape(n, w, -1)
    return np.concatenate([b.mean(1), b.std(1)], 1).astype(np.float32)


def rule_score(f: np.ndarray) -> np.ndarray:
    """Simple rule, no model: the largest deviation of a sensor mean from its calibration level."""
    return np.abs(f[:, :N_SENSORS]).max(1)


def summary(scores: np.ndarray, per_sensor: np.ndarray, threshold: float, budget: int) -> dict:
    """The bytes the MCU keeps for the next tap: a header and one uint8 per block of windows (the block's largest score).
    Blocks get longer when the period is long, so the summary never exceeds the budget."""
    slots = budget - HEADER_BYTES
    n = len(scores)
    per_block = max(1, -(-n // slots))  # ceiling division
    q = np.minimum(255, np.round(scores / threshold * Q_THRESHOLD)).astype(np.uint8)
    blocks = np.array([q[i:i + per_block].max() for i in range(0, n, per_block)], np.uint8)
    flagged = np.flatnonzero(scores > threshold)
    top = int(per_sensor[flagged].sum(0).argmax()) if len(flagged) else 255
    return {"blocks": blocks, "per_block": per_block, "flagged": len(flagged),
            "first": int(flagged[0]) if len(flagged) else None, "last": int(flagged[-1]) if len(flagged) else None,
            "top_sensor": top, "bytes": HEADER_BYTES + len(blocks)}


def alarm_from_summary(s: dict) -> bool:
    return bool((s["blocks"] > Q_THRESHOLD).any())
