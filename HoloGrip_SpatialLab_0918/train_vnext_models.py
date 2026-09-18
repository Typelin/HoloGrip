from __future__ import annotations
import csv, json, math, hashlib, sys
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP = ROOT / "Apps" / "Song_Collection_COM"
sys.path.insert(0, str(APP))
from product_hit_and_zone import ZONE_MAP, ZONE_NAMES

LAB = ROOT / "HoloGrip_SpatialLab_0918"
OUT = LAB / "vnext_models"
OUT.mkdir(parents=True, exist_ok=True)

RAW = ROOT / "Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT = ROOT / "Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"

FEATURE_NAMES = [
    "max_mag","mean_mag","std_mag","energy","jerk_max",
    "max_local_ax","max_local_ay","max_local_az",
    "mean_local_ax","mean_local_ay","mean_local_az",
    "rot_angle_std","rot_angle_range",
    "center_r6_0","center_r6_1","center_r6_2","center_r6_3","center_r6_4","center_r6_5",
    "mean_r6_0","mean_r6_1","mean_r6_2","mean_r6_3","mean_r6_4","mean_r6_5",
    "std_r6_0","std_r6_1","std_r6_2","std_r6_3","std_r6_4","std_r6_5",
]

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def circ_mean_deg(vals):
    a = np.asarray(vals, float)
    r = np.deg2rad(a)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))))

def load_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def build_session(rows):
    by = {h: [r for r in rows if r["hand"] == h] for h in ("L","R")}
    out = {}
    for h, rr in by.items():
        t = np.array([float(r["song_time_ms"]) for r in rr])
        first = t < t.min() + 1000.0
        y0 = circ_mean_deg([float(r["yaw_deg"]) for i,r in enumerate(rr) if first[i]])
        p0 = float(np.median([float(r["pitch_deg"]) for i,r in enumerate(rr) if first[i]]))
        r0 = circ_mean_deg([float(r["roll_deg"]) for i,r in enumerate(rr) if first[i]])
        R0 = Rot.from_euler("ZYX", [y0,p0,r0], degrees=True).as_matrix()
        samples = []
        for r in rr:
            Rc = Rot.from_euler("ZYX",
                                [float(r["yaw_deg"]), float(r["pitch_deg"]), float(r["roll_deg"])],
                                degrees=True).as_matrix()
            Rrel = R0.T @ Rc
            a = np.array([float(r["ax_g"]), float(r["ay_g"]), float(r["az_g"])], float)
            alocal = R0.T @ Rc @ a
            samples.append({
                "t": float(r["song_time_ms"]),
                "mag": float(r["accel_magnitude_g"]),
                "alocal": alocal,
                "Rrel": Rrel,
            })
        out[h] = {"R0": R0, "zero_euler": [y0,p0,r0], "samples": samples}
    return out

def r6(M):
    return M[:, :2].reshape(-1)

def feature_from_window(samples, center_ms, half_ms=80.0):
    w = [s for s in samples if center_ms-half_ms <= s["t"] <= center_ms+half_ms]
    if not w:
        return None
    mag = np.asarray([s["mag"] for s in w], float)
    A = np.vstack([s["alocal"] for s in w])
    mats = np.stack([s["Rrel"] for s in w])
    near = min(w, key=lambda s: abs(s["t"] - center_ms))
    r6s = np.stack([r6(M) for M in mats])
    center6 = r6(near["Rrel"])
    mean6 = np.mean(r6s, axis=0)
    std6 = np.std(r6s, axis=0)
    angles = np.asarray([Rot.from_matrix(M).magnitude() for M in mats], float)
    return np.asarray([
        np.max(mag), np.mean(mag), np.std(mag), np.sum(np.abs(mag-1.0)),
        np.max(np.abs(np.diff(mag))) if len(mag) > 1 else 0.0,
        *np.max(np.abs(A), axis=0).tolist(),
        *np.mean(A, axis=0).tolist(),
        float(np.std(angles)), float(np.max(angles)-np.min(angles)),
        *center6.tolist(), *mean6.tolist(), *std6.tolist(),
    ], float)

rows = load_rows(RAW)
session = build_session(rows)
with GT.open(encoding="utf-8", newline="") as f:
    gt = list(csv.DictReader(f))

X=[]; y=[]; hands=[]; groups=[]
for g in gt:
    h=g["final_hand"]; d=g["drum"]
    if h not in ("L","R") or d not in ZONE_MAP:
        continue
    c=float(g["csv_center_ms"])
    ft=feature_from_window(session[h]["samples"], c)
    if ft is None:
        continue
    X.append(ft); y.append(ZONE_MAP[d]); hands.append(h); groups.append(int(c//4000))
X=np.asarray(X); y=np.asarray(y); hands=np.asarray(hands); groups=np.asarray(groups)

def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1800, random_state=42))
    ])

metrics={}
for h in ("L","R"):
    idx=np.where(hands==h)[0]
    cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    pred=np.full(len(idx), -1, int)
    for tr0,te0 in cv.split(X[idx], y[idx], groups[idx]):
        m=make_model(); m.fit(X[idx][tr0], y[idx][tr0]); pred[te0]=m.predict(X[idx][te0])
    metrics[h]={
        "n": int(len(idx)),
        "accuracy": float(accuracy_score(y[idx],pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[idx],pred)),
        "macro_f1": float(f1_score(y[idx],pred,average="macro")),
        "counts": {ZONE_NAMES[k]: int(np.sum(y[idx]==k)) for k in range(len(ZONE_NAMES))}
    }
    model=make_model(); model.fit(X[idx],y[idx])
    mp=OUT/f"hologrip_vnext_{h}_relative_rotation.joblib"
    joblib.dump(model,mp)

# combined CV summary from per-hand predictions
pred_all=np.full(len(y),-1,int)
for h in ("L","R"):
    idx=np.where(hands==h)[0]
    cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr0,te0 in cv.split(X[idx], y[idx], groups[idx]):
        m=make_model(); m.fit(X[idx][tr0],y[idx][tr0]); pred_all[idx[te0]]=m.predict(X[idx][te0])

meta={
    "version":"vNext-relative-rotation-0918",
    "trained_from":str(RAW),
    "ground_truth":str(GT),
    "feature_names":FEATURE_NAMES,
    "feature_count":len(FEATURE_NAMES),
    "zone_names":ZONE_NAMES,
    "training_zero_euler":{h:[float(x) for x in session[h]["zero_euler"]] for h in ("L","R")},
    "per_hand_cv":metrics,
    "combined_cv":{
        "n":int(len(y)),
        "accuracy":float(accuracy_score(y,pred_all)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred_all)),
        "macro_f1":float(f1_score(y,pred_all,average="macro"))
    },
    "model_sha256":{
        h:sha256(OUT/f"hologrip_vnext_{h}_relative_rotation.joblib") for h in ("L","R")
    }
}
(OUT/"vnext_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(meta,ensure_ascii=False,indent=2))
