from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

REFRACTORY_MS = 120.0

class FrameValidator:
    def __init__(self):
        self.last_sig = None
        self.sig_repeat = 0
        self.invalid = 0
        self.reasons = Counter()

    def check(self, p: dict) -> tuple[bool,str]:
        vals=[p["ax"],p["ay"],p["az"],p["yaw"],p["pitch"],p["roll"]]
        if not all(math.isfinite(v) for v in vals):
            return self._bad("non_finite")
        if all(abs(v)<1e-12 for v in vals):
            return self._bad("all_zero")

        acc_same=max(abs(p["ax"]-p["ay"]),abs(p["ay"]-p["az"]),abs(p["ax"]-p["az"]))<1e-6
        ang_same=max(abs(p["yaw"]-p["pitch"]),abs(p["pitch"]-p["roll"]),abs(p["yaw"]-p["roll"]))<1e-6
        scale_same=abs(p["yaw"]-p["ax"]*11.25)<0.02
        if acc_same and ang_same and scale_same and abs(p["ax"])>0.01:
            return self._bad("i2c_same_word")

        mag=math.sqrt(p["ax"]**2+p["ay"]**2+p["az"]**2)
        if mag>28.0:
            return self._bad("impossible_accel")

        sig=tuple(round(p[k],4) for k in ("ax","ay","az","yaw","pitch","roll"))
        if sig==self.last_sig:
            self.sig_repeat += 1
        else:
            self.last_sig=sig
            self.sig_repeat=1
        if self.sig_repeat>=5 and mag>2.3:
            return self._bad("stale_high_plateau")
        return True,""

    def _bad(self,reason):
        self.invalid += 1
        self.reasons[reason] += 1
        return False,reason

class RefractoryGate:
    def __init__(self, refractory_ms: float = REFRACTORY_MS):
        self.refractory_ms=float(refractory_ms)
        self.last_accept_ms=-1e18
        self.suppressed=0

    def reset(self):
        self.last_accept_ms=-1e18
        self.suppressed=0

    def accept(self, peak_ms: float) -> bool:
        if peak_ms-self.last_accept_ms < self.refractory_ms:
            self.suppressed += 1
            return False
        self.last_accept_ms=float(peak_ms)
        return True

def make_model(random_state=42):
    return Pipeline([
        ("scaler",StandardScaler()),
        ("mlp",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1800,random_state=random_state)),
    ])

def load_training_archive(path: Path):
    z=np.load(path,allow_pickle=True)
    return z["X"].astype(float), z["y"].astype(int), z["hand"].astype(str)

def adapt_models(
    archive_path: Path,
    calibration_records: list[dict],
    output_dir: Path,
):
    """
    calibration_records:
      [{"hand":"L","label_id":0,"feature":[31 floats]}, ...]
    Each calibration hit is appended ONCE. This is intentional: repeated
    oversampling was less robust in cross-session experiments.
    """
    output_dir.mkdir(parents=True,exist_ok=True)
    X,y,h=load_training_archive(archive_path)
    models={}
    summary={}
    for hand in ("L","R"):
        base_idx=np.where(h==hand)[0]
        cal=[r for r in calibration_records if r["hand"]==hand]
        Xparts=[X[base_idx]]
        yparts=[y[base_idx]]
        if cal:
            Xc=np.asarray([r["feature"] for r in cal],float)
            yc=np.asarray([int(r["label_id"]) for r in cal],int)
            Xparts.append(Xc); yparts.append(yc)
        Xt=np.vstack(Xparts); yt=np.concatenate(yparts)
        m=make_model(42); m.fit(Xt,yt)
        out=output_dir/f"adapted_{hand}.joblib"
        joblib.dump(m,out)
        models[hand]=m
        summary[hand]={
            "base_n":int(len(base_idx)),
            "calibration_n":int(len(cal)),
            "fit_n":int(len(yt)),
            "model_path":str(out),
        }
    return models,summary
