"""Heading-free drum classifier for zero-direction estimation.

A wrong drumhead-zero yaw multiplies every zero-relative rotation by Rz(-delta): pitch and roll of the
rotations and the 83 trajectory dims are unchanged. This classifier uses only those (83 + pitch/roll
of the 7 pose samples = 97 dims), so its output does not depend on the zero direction. Per setting
(full and each held-out day) one XGBoost per hand on GPU; pretests never trained.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import joblib
import xgboost as xgb
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_1027 as FB
L5, K = FB.L5, FB.K
OUT = HERE/'out/cache'


def yawfree(X):
    """83 trajectory dims + pitch/roll of the 7 poses + body-frame rotation of each pose relative to the
    hit pose (R_hit^-1 R_k, unchanged by a zero-yaw error since R_k -> Rz(-d) R_k)."""
    X = np.asarray(X, np.float32); n = len(X)
    r = R.from_rotvec(X[:, 83:104].reshape(-1, 3).astype(np.float64)*np.pi)
    e = r.as_euler('ZYX', degrees=True).reshape(n, 7, 3)
    hit = R.from_rotvec(np.repeat(X[:, 83:104].reshape(n, 7, 3)[:, 4:5], 7, 1).reshape(-1, 3).astype(np.float64)*np.pi)
    body = (hit.inv()*r).as_rotvec().reshape(n, 7, 3)[:, [0, 1, 2, 3, 5, 6]].reshape(n, 18)
    return np.hstack([X[:, :83], e[:, :, 1]/90, e[:, :, 2]/180, body]).astype(np.float32)


def train(rows):
    out = {}
    for h in 'LR':
        rs = [r for r in rows if r['hand'] == h]
        X = yawfree(np.stack([r['x'] for r in rs])); y = np.array([K.DRUMS.index(r['drum']) for r in rs])
        out[h] = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=.05, subsample=.85, colsample_bytree=.7,
                                   device='cuda', tree_method='hist', objective='multi:softprob', num_class=7).fit(X, y)
    return out


def main():
    rows, *_ = FB.setup()
    for tag in ['full']+FB.DAYS:
        f = OUT/f'yawfree_{tag}.joblib'
        if f.exists(): continue
        hold = set() if tag == 'full' else {tag}
        joblib.dump(train([r for r in rows if L5.day_of(r['session']) not in hold]), f); print('yawfree', tag, flush=True)


if __name__ == '__main__':
    main()
