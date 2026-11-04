# tap-memory: what a machine should remember between two phone readings

## The problem

Many machines and devices have no permanent connection. They are read on site: a technician holds a phone close (NFC) or connects for a moment (BLE), reads a few variables from the microcontroller, then leaves. Between two readings, the machine keeps running, and an anomaly can appear and disappear with no witness.

The question: **for the same number of bytes read at each visit, what should be kept so that the phone knows something happened?** The memory of a small dynamic NFC tag holds a few hundred bytes (512 for 4 Kbit).

Four strategies are compared:

1. **The last state**: mean and spread of each sensor over the last 10 seconds (32 bytes). This is what you get when you read "the current variables".
2. **The latest raw readings**: as many seconds of readings as the budget allows, and the phone does the analysis.
3. **A summary kept by the microcontroller, with a simple rule**: every 10 s, the largest deviation of one sensor from its normal level. The summary keeps the maximum of each block of windows, in one byte.
4. **The same summary, with a small INT8-quantized autoencoder**: it learns how the sensors move *together* when all is well, and its score rises when they no longer agree.

## Method

- **Data**: SKAB (Skoltech Anomaly Benchmark), an instrumented pump test bench. It provides 8 sensors at one reading per second (two accelerometers, current, pressure, two temperatures, voltage, flow), 34 experiments of about twenty minutes with labelled anomalies (valves, cavitation, imbalance…), and one anomaly-free recording.
- **Calibration at installation**: during its first 5 minutes, each machine learns its own levels. Each sensor is scaled by its level and spread during those minutes, and the alarm threshold is a factor times the largest score seen during calibration.
- **The factor**: it is chosen on the anomaly-free recording only, by replaying an installation every minute. It is the factor exceeded by about 10 % of those installations within the following 10 minutes.
- **Two readings per experiment**: the first 10 s before the anomaly starts (the preceding period is normal), the second at the end of the recording. In most experiments, the anomaly is over before the second reading.
- **The autoencoder**: it takes 16 inputs (mean and spread of the 8 sensors over 10 s). It is trained on the first half of the anomaly-free recording, exported to ONNX, then quantized to INT8.
- **The summary** (`src/core.py`, function `summary`): an 8-byte header, then one byte per block of windows. The header holds the number of windows in alarm, the first and the last one, and the sensor most involved. When the period gets longer, the blocks get longer: the summary never exceeds the budget.

## Results

All figures below are written by `python -m src.run`.

<!-- results -->
33 experiments out of 34; 33 periods with an anomaly (median duration 9.7 min) and 33 normal periods.

| what the phone reads | bytes | anomalies seen | of which ended before the reading | of which ongoing | false alarms | median onset error |
|---|---:|---:|---:|---:|---:|---:|
| last state (mean and spread over the last 10 seconds) | 32 | 2 of 33 | 2 of 31 | 0 of 2 | 0 of 33 | – |
| latest raw readings | 64 | 2 of 33 | 2 of 31 | 0 of 2 | 0 of 33 | – |
| latest raw readings | 128 | 1 of 33 | 1 of 31 | 0 of 2 | 0 of 33 | – |
| latest raw readings | 256 | 1 of 33 | 1 of 31 | 0 of 2 | 0 of 33 | – |
| latest raw readings | 512 | 2 of 33 | 2 of 31 | 0 of 2 | 0 of 33 | – |
| summary, simple rule (largest deviation of one sensor) | 61 | 18 of 33 | 17 of 31 | 1 of 2 | 2 of 33 | 50.0 s |
| summary, simple rule (largest deviation of one sensor) | 84 | 18 of 33 | 17 of 31 | 1 of 2 | 2 of 33 | 50.0 s |
| summary, INT8 autoencoder | 61 | 19 of 33 | 19 of 31 | 0 of 2 | 2 of 33 | 50.0 s |
| summary, INT8 autoencoder | 84 | 19 of 33 | 19 of 31 | 0 of 2 | 2 of 33 | 50.0 s |

Threshold of each machine: factor × the largest score seen during its 5 minutes of calibration (autoencoder ×5.848, rule ×3.211). The factors are chosen on 64 installations replayed in the anomaly-free recording, so that about 10 % of them raise a false alarm within 10 minutes.

Full trade-off, varying the factor (anomalies seen / false alarms, on the same periods):

| factor | 1 | 1.5 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| summary, autoencoder | 33 / 25 | 31 / 11 | 29 / 7 | 27 / 3 | 24 / 3 | 19 / 2 | 15 / 0 | 13 / 0 | 11 / 0 |
| summary, simple rule | 31 / 24 | 27 / 9 | 26 / 4 | 18 / 2 | 16 / 1 | 11 / 0 | 9 / 0 | 7 / 0 | 6 / 0 |
| last state | 16 / 11 | 9 / 3 | 7 / 1 | 3 / 1 | 2 / 1 | 2 / 0 | 1 / 0 | 1 / 0 | 1 / 0 |

Sensor most involved according to the summary (autoencoder, 512 bytes), by experiment series:

- other: vibration 1 (5), flow (1), thermocouple (1)
- valve1: flow (8), vibration 2 (2)
- valve2: flow (2)

Resolution of the summary as the period between two readings gets longer (seconds covered by one byte):

| budget | 1 hour | 1 day | 1 week |
|---:|---:|---:|---:|
| 64 bytes | 70 s | 1,550 s | 10,800 s |
| 128 bytes | 30 s | 720 s | 5,040 s |
| 256 bytes | 20 s | 350 s | 2,440 s |
| 512 bytes | 10 s | 180 s | 1,200 s |

On AMD Ryzen 7 5825U with Radeon Graphics, one thread: a 10 s window costs 12.8 µs in FP32 and 17.3 µs in INT8 (356 parameters, 320 multiply-accumulates; ONNX file of 2,263 bytes in FP32, 5,820 in INT8). INT8 and FP32 make the same decision on 100.0 % of windows.
<!-- /results -->

**In short**: what the microcontroller keeps matters more than the model. The last state and the raw readings almost never show an anomaly that is already over, while a summary of less than 90 bytes shows most of them. At the chosen operating point, the autoencoder and the simple rule are almost even. Across the whole trade-off, the autoencoder sees more anomalies for the same number of false alarms: probably because it spots sensors that drift apart without any of them clearly leaving its range (a hypothesis, not checked sensor by sensor). For the valve experiments, the sensor named is most often the flow.

## Limitations

- **Short periods**: the experiments last about twenty minutes, so the byte budget does not constrain the summary. The resolution for periods from one hour to one week is computed, not tested.
- **One bench, one pump**: about thirty periods with an anomaly is few, and the share of anomalies seen varies a lot with the chosen factor (see the trade-off curve).
- **One experiment left out**: its anomaly starts during the 5 minutes of calibration.
- **No hardware tested**: no NFC tag, no radio module, no microcontroller. Cost measured with ONNX Runtime on a PC processor. On this processor, the INT8 version is not faster than FP32 for such a small network, and its ONNX file is even larger (the quantization nodes weigh more than the weights).
- **False alarms underestimated**: the normal periods last only a few minutes. Over a day between two readings, there would be more chances of a false alarm.

## Reproduce

```
python -m venv .venv && .venv/Scripts/activate        # source .venv/bin/activate on Linux
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m src.fetch     # downloads SKAB (about 5 MB)
python -m src.run       # trains, replays the readings, writes results/ and docs/
python -m pytest -q
```

The demo page (`docs/index.html`) reads `docs/results.json` and `docs/series.json`.

## Data

Katser, I. D., Kozitsin, V. O. *Skoltech Anomaly Benchmark (SKAB)*, Kaggle, 2020, https://github.com/waico/SKAB. GPL-3.0 licence. The repository does not contain the data: `src/fetch.py` downloads it.
