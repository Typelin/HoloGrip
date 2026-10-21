"""Light cymbal hits are often below the 2.1 g detector threshold. Rescue rule (code, not training):
detect at a lower threshold; peaks >= 2.1 g are output as usual; weaker peaks are output only if the
model (after the per-take offset) calls them a cymbal with score >= theta. Checked on the 14 pretest
cells and on the held-out song 04 full pipeline (extra outputs vs MIDI)."""
import json, itertools
import numpy as np, joblib
from scipy.optimize import linear_sum_assignment
import common as K, cells as Q, h_common as H
import full_report as F
import soft_hit_experiment as S
import venue_free_runtime_1020 as rt
import causal_hit_detector_1016 as detector
import seventh_core as core
from h3_proto_grid import make_post

CYM = {'Hi-Hat', 'Crash', 'Ride'}


def items_at(raw, cfg, thr):
    hands = rt.world.read_raw(raw); prep = rt.world.base.legacy.prepare(hands)
    peaks, _ = detector.detect(hands, dict(cfg, threshold_g=thr)); out = []
    for p in peaks:
        s = prep[p['hand']]['rows'][p['available_index']]
        x = rt.feature(prep, p, (s['absolute_yaw']-s['yaw']+180) % 360-180, s['absolute_pitch']-s['pitch'])
        if x is not None: out.append(dict(t=float(p['peak_ms']), hand=p['hand'], g=float(p['peak_g']), x=x))
    return out


def run(com, post, items, theta):
    P = []
    for h in 'LR':
        it = sorted([c for c in items if c['hand'] == h], key=lambda c: c['t'])
        strong = [c for c in it if c['g'] >= 2.1]
        X = np.stack([c['x'] for c in it]); p = post(com, h, it, X)
        for c, q in zip(it, p):
            d = K.DRUMS[int(q.argmax())]
            if c['g'] >= 2.1 or (d in CYM and q.max() >= theta): P.append(dict(t=c['t'], hand=h, drum=d))
    return P


def main():
    _, cfg = core.load_components(); com = joblib.load(K.OUT/'h_best_com.joblib')
    train, hold, tests = H.load(with_timing=False)
    tm = {(h, d): np.mean([r['x'][H.ROT] for r in train if r['hand'] == h and r['drum'] == d], 0) for h in 'LR' for d in K.DRUMS
          if any(r['hand'] == h and r['drum'] == d for r in train)}
    post = make_post(tm, 1.0, 0.7, 20, 2)
    raws = {k: S.C['tests'][k]['raw'] for k in K.TESTS}; raws['song04'] = S.C['sessions']['seventh_04']['raw']
    used, excl = F.truth_of('seventh_04')
    for thr in (2.1, 1.9, 1.7):
        its = {k: items_at(r, cfg, thr) for k, r in raws.items()}
        for theta in ((0.0,) if thr == 2.1 else (0.6, 0.8, 0.9)):
            preds = {k: run(com, post, its[k], theta) for k in K.TESTS}
            r = Q.score_cells(preds); res, _ = F.score(used, excl, run(com, post, its['song04'], theta))
            print(f'thr{thr} theta{theta}', Q.fmt(r), f"| song04 correct {res.get('correct',0)}/{res['midi']} extra {res['extra']} missed {res.get('missed',0)}", flush=True)


if __name__ == '__main__':
    main()
