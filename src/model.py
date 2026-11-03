"""A tiny autoencoder that learns how the pump's sensors move together when everything is normal.

It reconstructs the 16 window features (mean and spread of 8 sensors); a large reconstruction error means the sensors
no longer agree with each other the way they usually do. Exported to ONNX and quantized to INT8 for the MCU.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from .fetch import ROOT

MODELS = ROOT / "models"
N_IN = 16


def net() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(N_IN, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4), torch.nn.ReLU(),
                               torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, N_IN))


def train(x: np.ndarray, epochs: int = 1500) -> torch.nn.Module:
    torch.manual_seed(0)
    m = net()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    t = torch.from_numpy(x)
    for _ in range(epochs):  # the training set is small (a few thousand windows): full batch
        opt.zero_grad()
        loss = ((m(t) - t) ** 2).mean()
        loss.backward()
        opt.step()
    return m.eval()


class _Calib(CalibrationDataReader):
    def __init__(self, x: np.ndarray):
        self.rows = iter(x[i:i + 256] for i in range(0, len(x), 256))

    def get_next(self):
        b = next(self.rows, None)
        return None if b is None else {"f": b}


def export(m: torch.nn.Module, calib: np.ndarray, name: str) -> tuple[Path, Path]:
    MODELS.mkdir(exist_ok=True)
    fp32, int8, pre = (MODELS / f"{name}_{k}.onnx" for k in ("fp32", "int8", "pre"))
    torch.onnx.export(m, torch.zeros(1, N_IN), str(fp32), input_names=["f"], output_names=["r"],
                      dynamic_axes={"f": {0: "n"}, "r": {0: "n"}}, opset_version=17, dynamo=False)
    quant_pre_process(str(fp32), str(pre))
    # reduce_range: 7-bit weights avoid the U8S8 int16 saturation of x86 CPUs without VNNI
    quantize_static(str(pre), str(int8), _Calib(calib), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, reduce_range=True)
    pre.unlink()
    return fp32, int8


def session(path: Path) -> ort.InferenceSession:
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def errors(sess: ort.InferenceSession, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score of each window (mean squared reconstruction error) and the error of each sensor (mean + spread)."""
    r = sess.run(None, {"f": f.astype(np.float32)})[0]
    e = (r - f) ** 2
    return e.mean(1), e[:, :8] + e[:, 8:]
