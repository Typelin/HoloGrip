"""Cache trained committees + accepted pretest hits for fast layout-calibration experiments.

For each setting (full = all five days; R4..R8 = that day held out entirely) store:
  com_raw, com_rc (1 seed each, raw rows / re-centred rows), drum means of re-centred rows (tm),
  and for every pretest recording the gate-accepted hits per hand (t, x, raw committee probabilities).
Pretests are never trained.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mix_1027 as MX
RC, FB, L5, K, H = MX.RC, MX.FB, MX.L5, MX.K, MX.H
OUT = FB.OUT/'cache'; OUT.mkdir(exist_ok=True)


def accepted(name, gate):
    out = {}
    for h in 'LR':
        cs = sorted([c for c in H.CAND[name] if c['hand'] == h], key=lambda c: c['t'])
        ph = gate.predict_proba(np.stack([H.feats(c) for c in cs]))[:, 1]; acc = [c for c, q in zip(cs, ph) if q >= .4]
        out[h] = dict(t=np.array([c['t'] for c in acc]), X=np.stack([c['x'] for c in acc]).astype(np.float32))
    return out


def main():
    rows, names, Lb, T8, R8 = FB.setup()
    pre = [n for v in FB.PRETESTS.values() for n in v]
    for tag in ['full']+FB.DAYS:
        f = OUT/f'{tag}.joblib'
        if f.exists(): continue
        hold = set() if tag == 'full' else {tag}
        tr = [r for r in rows if L5.day_of(r['session']) not in hold]; rc = RC.recenter(tr); tm = MX.means(rc); gate = MX.gate_for(names, Lb, hold)
        com_raw = K.train(tr, {}, seeds=(0,)); com_rc = K.train(rc, {}, seeds=(0,))
        items = {n: accepted(n, gate) for n in pre if tag == 'full' or L5.day_of(n) == tag}
        for n, d in items.items():
            for h, v in d.items(): v['p_raw'] = K.proba(com_raw, h, v['X']); v['p_rc'] = K.proba(com_rc, h, v['X'])
        joblib.dump(dict(com_raw=com_raw, com_rc=com_rc, tm=tm, items=items, truth={n: L5.TRUTH[n] for n in items},
                         tm_raw=MX.means(tr)), f)
        print('cached', tag, len(tr), flush=True)


if __name__ == '__main__':
    main()
