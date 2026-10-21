"""Targeted style augmentation: pretests play Ride/Crash with a lower arm than songs. Add copies of
cymbal training hits whose zero-relative pitch is lowered by U(lo, hi) degrees (drumhead-zero pitch
raised). Then the best pipeline (prototype per-take offset + cymbal light-hit rescue at 1.7 g).
Checked on 14 pretest cells, held-out song 04 and single-drum takes."""
import itertools
import numpy as np, joblib
import common as K, cells as Q, h_common as H
import full_report as F
from h2_style_offset import train_w
from h3_proto_grid import make_post
from h5_cymbal_rescue import items_at
from phase3_aug import x_with_zero
import soft_hit_experiment as S
import seventh_core as core


def augment(train, drums, lo, hi, copies, seed=0):
    rng = np.random.default_rng(seed); out = list(train)
    for r in train:
        if r['drum'] in drums:
            for _ in range(copies): out.append(dict(r, x=x_with_zero(r, 0., rng.uniform(lo, hi))))
    return out


def main():
    train, hold, tests = H.load(with_timing=False); _, cfg = core.load_components()
    raws = {k: S.C['tests'][k]['raw'] for k in K.TESTS}; raws['song04'] = S.C['sessions']['seventh_04']['raw']
    singles = ('seventh_06', 'seventh_09', 'sixth_S06', 'sixth_S07')
    for s in singles: raws[s] = S.C['sessions'][s]['raw']
    its = {k: items_at(r, cfg, 1.7) for k, r in raws.items()}
    used, excl = F.truth_of('seventh_04')
    for drums, lo, hi, copies in [(set(), 0, 0, 0), ({'Ride'}, 5, 25, 3), ({'Ride', 'Crash'}, 5, 25, 3), ({'Ride', 'Crash'}, 10, 30, 3), ({'Ride', 'Crash', 'Hi-Hat'}, 5, 20, 2)]:
        tr = augment(train, drums, lo, hi, copies); com = train_w(tr, lambda r: r['x'], lambda r: 1.)
        tm = {(h, d): np.mean([r['x'][H.ROT] for r in train if r['hand'] == h and r['drum'] == d], 0) for h in 'LR' for d in K.DRUMS
              if any(r['hand'] == h and r['drum'] == d for r in train)}
        post = make_post(tm, 1.0, 0.7, 20, 2); out = {}
        for k, items in its.items():
            P = []
            for h in 'LR':
                it = sorted([c for c in items if c['hand'] == h], key=lambda c: c['t'])
                if not it: continue
                p = post(com, h, it, np.stack([c['x'] for c in it]))
                for c, q in zip(it, p):
                    d = K.DRUMS[int(q.argmax())]
                    if c['g'] >= 2.1 or (d in ('Hi-Hat', 'Crash', 'Ride') and q.max() >= .6): P.append(dict(t=c['t'], hand=h, drum=d))
            out[k] = P
        r = Q.score_cells({k: out[k] for k in K.TESTS}); res, _ = F.score(used, excl, out['song04'])
        sg = []
        for s in singles:
            u, e = F.truth_of(s); rr, _ = F.score(u, e, out[s]); sg.append(f"{rr.get('correct',0)}/{rr['midi']}")
        joblib.dump(com, K.OUT/f"h7_com_{'_'.join(sorted(drums)) or 'none'}_{lo}_{hi}.joblib")
        print(sorted(drums), lo, hi, copies, Q.fmt(r), f"| song04 {res.get('correct',0)}/{res['midi']} extra {res['extra']} | singles {' '.join(sg)}", flush=True)


if __name__ == '__main__':
    main()
