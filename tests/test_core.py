import numpy as np

from src import core


def test_standardize_floors_a_constant_sensor():
    ref = np.column_stack([np.random.default_rng(0).normal(5, 1, 300), np.full(300, 32.0)]).astype(np.float32)
    z = core.standardize(np.array([[5.0, 32.64]], np.float32), ref)
    assert abs(z[0, 1] - 1.0) < 1e-5  # spread floored at 2 % of 32: a 0.64 change is one unit, not infinity


def test_features_windows_and_short_input():
    z = np.arange(25, dtype=np.float32)[:, None].repeat(8, 1)
    f = core.features(z)
    assert f.shape == (2, 16) and f[0, 0] == 4.5  # two full windows, the last 5 samples dropped
    assert core.features(z[:4]).shape == (1, 16)  # shorter than a window: one window of what there is


def test_summary_fits_budget_and_keeps_a_past_alarm():
    scores = np.full(1000, 0.1)
    scores[300] = 5.0  # one alarm long before the tap
    per = np.zeros((1000, 8))
    per[300, 7] = 1.0
    s = core.summary(scores, per, threshold=1.0, budget=64)
    assert s["bytes"] <= 64 and core.alarm_from_summary(s)
    assert s["first"] == 300 and s["top_sensor"] == 7 and s["per_block"] == -(-1000 // (64 - core.HEADER_BYTES))
    assert not core.alarm_from_summary(core.summary(np.full(10, 0.5), np.zeros((10, 8)), 1.0, 64))
