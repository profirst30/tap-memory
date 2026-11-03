"""Whole pipeline: learn the pump's normal behaviour, replay two phone taps per experiment, and compare what each
storage strategy lets the phone conclude for the same number of bytes.

    python -m src.run     # writes results/results.json, docs/results.json, docs/series.json, README table

Every number in the README and on the demo page comes from this script.
"""
import json
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

from . import core, model
from .fetch import ROOT, SENSORS, experiments, normal, read

RESULTS = ROOT / "results" / "results.json"
BUDGETS = [64, 128, 256, 512]  # bytes; 512 is the user memory of the smallest dynamic NFC tag of the ST25DV family (4 Kbit)
NORMAL_CHUNK = 600  # seconds per normal test period cut from the anomaly-free recording
SENSOR_FR = ["vibration 1", "vibration 2", "courant", "pression", "température", "thermocouple", "tension", "débit"]
LONG = {"1 heure": 3600, "1 jour": 86400, "1 semaine": 604800}
FA_TARGET = 0.1  # share of replayed installations allowed one false alarm in the 10 minutes that follow
CURVE = [1, 1.5, 2, 3, 4, 6, 8, 12, 16]  # threshold factors shown on the trade-off curve


def cpu_name() -> str:
    if sys.platform == "win32":
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                           capture_output=True, text=True)
        return r.stdout.strip() or platform.processor()
    return platform.processor() or platform.machine()


def bench_us(fn, n: int = 500) -> float:
    for _ in range(20):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return float(np.median(t) * 1e6)


def main() -> None:
    xn, _ = read(normal())
    zn = core.standardize(xn, xn[:core.CAL])
    n = len(zn)
    train_end = int(0.5 * n)

    exps = []
    for p in experiments():
        x, a = read(p)
        onset = int(np.flatnonzero(a)[0])
        if onset < core.CAL + 2 * core.W:  # the anomaly starts during calibration: the "normal" reference is wrong
            continue
        exps.append({"name": f"{p.parent.name}/{p.stem}", "group": p.parent.name, "z": core.standardize(x, x[:core.CAL]),
                     "a": a, "onset": onset})

    # Training: the first half of the anomaly-free recording only.
    train = core.features(zn[core.CAL:train_end])
    m = model.train(train)
    fp32, int8 = model.export(m, train, "pump_ae")
    s8, s32 = model.session(int8), model.session(fp32)

    # Thresholds learned by each machine during its own calibration minutes: factor x the largest score seen then.
    # The factor is chosen on the second half of the anomaly-free recording only, by replaying installations every
    # minute: the factor exceeded by FA_TARGET of these installations (so about 1 in 10 raises a false alarm).
    def scores(z: np.ndarray) -> dict:
        f = core.features(z)
        return {"ae": model.errors(s8, f)[0], "rule": core.rule_score(f)}

    ratios = {"ae": [], "rule": []}
    for start in range(train_end, n - core.CAL - NORMAL_CHUNK + 1, 60):
        z = core.standardize(xn[start:start + core.CAL + NORMAL_CHUNK], xn[start:start + core.CAL])
        cal, after = scores(z[:core.CAL]), scores(z[core.CAL:])
        for k in ratios:
            ratios[k].append(float(after[k].max() / cal[k].max()))
    factor = {k: float(np.quantile(v, 1 - FA_TARGET)) for k, v in ratios.items()}
    for e in exps:
        cal = scores(e["z"][:core.CAL])
        e["thr"] = {k: factor[k] * float(cal[k].max()) for k in factor}

    # Periods between taps: [calibration end, 10 s before the anomaly] is normal, [that tap, end of file] holds the anomaly.
    periods = []
    for e in exps:
        tap_a = e["onset"] - core.W
        periods.append({"exp": e, "a": core.CAL, "b": tap_a, "anomalous": False})
        periods.append({"exp": e, "a": tap_a, "b": len(e["z"]), "anomalous": True})

    # Trade-off curve: the same periods with other threshold factors (whole-period summaries and the last state).
    curve = {"factors": CURVE, "summary_ae": [], "summary_rule": [], "last_state": []}
    pre = []
    for p in periods:
        e = p["exp"]
        sc = scores(e["z"][p["a"]:p["b"]])
        cal = scores(e["z"][:core.CAL])
        last = model.errors(s8, core.features(e["z"][p["b"] - core.W:p["b"]]))[0].max()
        pre.append((p["anomalous"], {k: sc[k].max() / cal[k].max() for k in sc}, last / cal["ae"].max()))
    for c in CURVE:
        for key, get in (("summary_ae", lambda r: r[1]["ae"]), ("summary_rule", lambda r: r[1]["rule"]), ("last_state", lambda r: r[2])):
            curve[key].append({"detected": sum(1 for r in pre if r[0] and get(r) > c),
                               "false_alarms": sum(1 for r in pre if not r[0] and get(r) > c)})

    methods = ["last_state"] + [f"{k}_{b}" for k in ("raw_tail", "summary_rule", "summary_ae") for b in BUDGETS]
    rows = {k: [] for k in methods}
    agree = []
    series = {}
    for p in periods:
        e, z = p["exp"], p["exp"]["z"][p["a"]:p["b"]]
        thr_e = e["thr"]
        lab = e["a"][p["a"]:p["b"]]
        ended = bool(p["anomalous"] and lab[-1] == 0)
        f = core.features(z)
        ae, per_sensor = model.errors(s8, f)
        ae32, _ = model.errors(s32, f)
        agree.append(float(((ae > thr_e["ae"]) == (ae32 > thr_e["ae"])).mean()))
        rule = core.rule_score(f)
        base = {"name": e["name"], "group": e["group"], "anomalous": p["anomalous"], "ended": ended, "seconds": int(p["b"] - p["a"])}
        last = model.errors(s8, core.features(z[-core.W:]))[0]
        rows["last_state"].append(base | {"alarm": bool(last.max() > thr_e["ae"]), "bytes": core.STATE_BYTES})
        onset_s = int(np.flatnonzero(lab)[0]) if p["anomalous"] else None
        for b in BUDGETS:
            k = b // core.SAMPLE_BYTES
            tail = model.errors(s8, core.features(z[-k:]))[0]
            rows[f"raw_tail_{b}"].append(base | {"alarm": bool(tail.max() > thr_e["ae"]), "bytes": k * core.SAMPLE_BYTES, "seconds_kept": k})
            for kind, sc in (("rule", rule), ("ae", ae)):
                s = core.summary(sc, per_sensor, thr_e[kind], b)
                r = base | {"alarm": core.alarm_from_summary(s), "bytes": s["bytes"], "per_block": s["per_block"],
                            "top_sensor": s["top_sensor"] if kind == "ae" else None}
                if p["anomalous"] and s["first"] is not None:
                    r["onset_error_s"] = abs(s["first"] * core.W - onset_s)
                    r["flagged_s"] = s["flagged"] * core.W
                    r["true_s"] = int(lab.sum())
                rows[f"summary_{kind}_{b}"].append(r)
        if p["anomalous"]:
            series[e["name"]] = {"tap": int(p["a"] // core.W), "onset": int(e["onset"] // core.W), "threshold": round(thr_e["ae"], 4),
                                 "ae": [round(float(v), 3) for v in model.errors(s8, core.features(e["z"]))[0]],
                                 "label": [int(v) for v in core.features(e["a"][:, None].astype(np.float32))[:, 0] > 0.5]}

    def pool(rs: list[dict]) -> dict:
        an = [r for r in rs if r["anomalous"]]
        no = [r for r in rs if not r["anomalous"]]
        ended = [r for r in an if r["ended"]]
        ongoing = [r for r in an if not r["ended"]]
        on = [r["onset_error_s"] for r in an if r["alarm"] and "onset_error_s" in r]
        return {"bytes_max": max(r["bytes"] for r in rs),
                "detected": sum(r["alarm"] for r in an), "anomalous": len(an),
                "detected_ended": sum(r["alarm"] for r in ended), "ended": len(ended),
                "detected_ongoing": sum(r["alarm"] for r in ongoing), "ongoing": len(ongoing),
                "false_alarms": sum(r["alarm"] for r in no), "normal": len(no),
                "onset_error_median_s": float(np.median(on)) if on else None}

    table = {k: pool(v) for k, v in rows.items()}
    top = {}
    for r in rows["summary_ae_512"]:
        if r["anomalous"] and r["alarm"] and r["top_sensor"] is not None and r["top_sensor"] < 8:
            g = top.setdefault(r["group"], {})
            g[SENSOR_FR[r["top_sensor"]]] = g.get(SENSOR_FR[r["top_sensor"]], 0) + 1
    long = {}
    for b in BUDGETS:
        slots = b - core.HEADER_BYTES
        long[str(b)] = {name: -(-(sec // core.W) // slots) * core.W for name, sec in LONG.items()}  # seconds per block

    lin = [l for l in m if hasattr(l, "in_features")]
    one = train[:1]
    cost = {"fp32_us": round(bench_us(lambda: s32.run(None, {"f": one})), 1), "int8_us": round(bench_us(lambda: s8.run(None, {"f": one})), 1),
            "parameters": sum(p.numel() for p in m.parameters()), "macs": sum(l.in_features * l.out_features for l in lin),
            "fp32_bytes": fp32.stat().st_size, "int8_bytes": int8.stat().st_size,
            "int8_fp32_window_agreement": round(float(np.mean(agree)), 4)}
    an_periods = [p for p in periods if p["anomalous"]]
    out = {"machine": {"cpu": cpu_name(), "os": platform.platform(), "onnxruntime": ort.__version__},
           "settings": {"calibration_s": core.CAL, "window_s": core.W, "budgets": BUDGETS, "header_bytes": core.HEADER_BYTES,
                        "sample_bytes": core.SAMPLE_BYTES, "state_bytes": core.STATE_BYTES, "normal_chunk_s": NORMAL_CHUNK,
                        "sensors": SENSORS},
           "data": {"experiments_used": len(exps), "experiments_total": len(experiments()), "train_windows": int(len(train)),
                    "installations_replayed": len(ratios["ae"]), "anomalous_periods": len(an_periods),
                    "normal_periods": len(periods) - len(an_periods),
                    "period_s_median": float(np.median([p["b"] - p["a"] for p in an_periods]))},
           "threshold_factor": {k: round(v, 3) for k, v in factor.items()},
           "methods": table, "curve": curve, "fa_target": FA_TARGET, "top_sensor": top, "seconds_per_block": long, "cost": cost}
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(RESULTS, ROOT / "docs" / "results.json")
    (ROOT / "docs" / "series.json").write_text(json.dumps({"series": series}, separators=(",", ":")), encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        text = re.sub(r"<!-- results -->.*<!-- /results -->", lambda _: f"<!-- results -->\n{md(out)}\n<!-- /results -->",
                      readme.read_text(encoding="utf-8"), flags=re.S)
        readme.write_text(text, encoding="utf-8")
    print(md(out))


LABEL = {"last_state": "dernier état (moyenne et dispersion des 10 dernières secondes)", "raw_tail": "dernières mesures brutes",
         "summary_rule": "résumé, règle simple (plus grand écart d'un capteur)", "summary_ae": "résumé, auto-encodeur INT8"}


def fr(x) -> str:
    return "–" if x is None else f"{x:,}".replace(",", " ").replace(".", ",")


def md(out: dict) -> str:
    d, M = out["data"], out["methods"]
    L = [f"{d['experiments_used']} expériences sur {d['experiments_total']} ; {d['anomalous_periods']} périodes avec anomalie "
         f"(durée médiane {fr(round(d['period_s_median'] / 60, 1))} min) et {d['normal_periods']} périodes normales.", "",
         "| ce que le téléphone lit | octets | anomalies vues | dont anomalie finie avant le passage | dont anomalie en cours | fausses alarmes | erreur médiane sur le début |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for k, r in M.items():
        kind = k.rsplit("_", 1)[0] if k != "last_state" else k
        if kind.startswith("summary") and not (k.endswith("_64") or k.endswith("_512")):
            continue  # for periods of about ten minutes the summary uses the same bytes at 128, 256 and 512
        L.append(f"| {LABEL[kind]} | {r['bytes_max']} | {r['detected']} sur {r['anomalous']} | {r['detected_ended']} sur {r['ended']} | "
                 f"{r['detected_ongoing']} sur {r['ongoing']} | {r['false_alarms']} sur {r['normal']} | "
                 f"{fr(r['onset_error_median_s'])}{' s' if r['onset_error_median_s'] is not None else ''} |")
    L += ["", f"Seuil de chaque machine : facteur × le plus grand score vu pendant ses {out['settings']['calibration_s'] // 60} minutes de calibration "
          f"(auto-encodeur ×{fr(out['threshold_factor']['ae'])}, règle ×{fr(out['threshold_factor']['rule'])}). Les facteurs sont choisis sur "
          f"{out['data']['installations_replayed']} installations rejouées dans l'enregistrement sans anomalie, pour qu'environ "
          f"{fr(round(100 * out['fa_target']))} % d'entre elles donnent une fausse alarme dans les 10 minutes.", "",
          "Compromis complet, en changeant le facteur (anomalies vues / fausses alarmes, sur les mêmes périodes) :", "",
          "| facteur | " + " | ".join(str(fr(f)) for f in out["curve"]["factors"]) + " |", "|---|" + "---:|" * len(out["curve"]["factors"])]
    for key, name in (("summary_ae", "résumé, auto-encodeur"), ("summary_rule", "résumé, règle simple"), ("last_state", "dernier état")):
        L.append(f"| {name} | " + " | ".join(f"{c['detected']} / {c['false_alarms']}" for c in out["curve"][key]) + " |")
    L += ["", "Capteur le plus en cause selon le résumé (auto-encodeur, 512 octets), par série d'expériences :", ""]
    for g, c in out["top_sensor"].items():
        L.append(f"- {g} : " + ", ".join(f"{k} ({v})" for k, v in sorted(c.items(), key=lambda kv: -kv[1])))
    L += ["", "Résolution du résumé quand la période entre deux passages s'allonge (secondes couvertes par un octet) :", "",
          "| budget | " + " | ".join(LONG) + " |", "|---:|" + "---:|" * len(LONG)]
    for b, v in out["seconds_per_block"].items():
        L.append(f"| {b} octets | " + " | ".join(fr(v[name]) + " s" for name in LONG) + " |")
    c = out["cost"]
    L += ["", f"Sur {out['machine']['cpu']}, un thread : une fenêtre de 10 s coûte {fr(c['fp32_us'])} µs en FP32 et {fr(c['int8_us'])} µs en INT8 "
          f"({c['parameters']} paramètres, {c['macs']} multiplications-additions ; fichier ONNX de {fr(c['fp32_bytes'])} octets en FP32, "
          f"{fr(c['int8_bytes'])} en INT8). INT8 et FP32 prennent la même décision sur {fr(round(100 * c['int8_fp32_window_agreement'], 2))} % des fenêtres."]
    return "\n".join(L)


if __name__ == "__main__":
    main()
