"""Does the zero-relative yaw drift inside one recording (no magnetometer)? Labelled song / four-drum takes:
per hand, least-squares fit  yaw = a[drum] + b * minutes  (one offset per drum, one common slope b) and the same
for pitch. Read-only; poses from the field feature extractor with the zero-direction guard off."""
import sys
import numpy as np
import pose_r12_1110 as PR
from multiprocessing import Pool
import csv, joblib
core = PR.core; HERE = PR.HERE


def work(job):
    nm, raw, truth = job
    b = joblib.load(HERE/'out/model_r10c.joblib'); core.runtime.attach_boosters(b); b.pop('_zero_estimator', None); config = core.load_components()[1]
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(b, config, snare_anchor=5); rt = engine.runtime; rec = []; orig = rt._classify
    def spy(hand, x):
        rec.append(core.runtime._yaw_pitch(x).tolist()); return orig(hand, x)
    rt._classify = spy; out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]; n0 = len(rec)
        p, _ = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        for q, yp in zip(p, rec[n0:]): out.append((q['hand'], q['peak_ms'], yp[0], yp[1]))
    res = {}
    for h in 'LR':
        rows = []
        for g in truth:
            if g['hand'] != h: continue
            near = [o for o in out if o[0] == h and abs(o[1]-g['t']) <= 60]
            if near: o = min(near, key=lambda o: abs(o[1]-g['t'])); rows.append((g['drum'], o[1]/60000, o[2], o[3]))
        drums = sorted({r[0] for r in rows if sum(x[0] == r[0] for x in rows) >= 8}); rows = [r for r in rows if r[0] in drums]
        if len(rows) < 40: continue
        t = np.array([r[1] for r in rows]); A = np.zeros((len(rows), len(drums)+1)); A[:, -1] = t
        for i, r in enumerate(rows): A[i, drums.index(r[0])] = 1
        yaw = np.unwrap(np.radians([r[2] for r in rows]))*180/np.pi   # crude; per-drum offsets absorb wraps of far drums
        out_h = {}
        for name, y in (('yaw', np.array([r[2] for r in rows])), ('pitch', np.array([r[3] for r in rows]))):
            if name == 'yaw':   # centre each drum circularly before fitting
                for d in drums:
                    m = np.array([r[0] == d for r in rows]); c = np.degrees(np.angle(np.exp(1j*np.radians(y[m])).mean())); y[m] = (y[m]-c+180) % 360-180
            coef, *_ = np.linalg.lstsq(A, y, rcond=None); resid = y-A@coef
            out_h[name] = (float(coef[-1]), float(resid.std()))
        res[h] = dict(n=len(rows), minutes=float(t.max()-t.min()), **out_h)
    return nm, res


if __name__ == '__main__':
    J = [j for j in PR.jobs() if not j[0].startswith('pretest') and ('歌曲' in j[0] or '四鼓' in j[0])]
    with Pool(4) as pool:
        for nm, res in pool.imap(work, J):
            for h, r in res.items():
                print(f"{nm:14s} {h} n={r['n']:4d} {r['minutes']:.1f} min  yaw slope {r['yaw'][0]:+6.1f} deg/min (total {r['yaw'][0]*r['minutes']:+6.1f} deg, scatter {r['yaw'][1]:.1f})   pitch slope {r['pitch'][0]:+5.1f} deg/min (total {r['pitch'][0]*r['minutes']:+5.1f})", flush=True)
