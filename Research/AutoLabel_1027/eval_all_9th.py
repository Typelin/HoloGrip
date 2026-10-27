"""Score every round 4-8 recording (pretests, songs, four-drum, single-hand) with the INSTALLED 9th package
model: hit gate + weighted committee + automatic anchor + per-drum offset + position prior + continuity,
in batch (same causal logic as the field engine; pretests are first checked against the sample-by-sample
replay). Everything except the pretests was used for training, so non-pretest scores are training-data
(resubstitution) scores. Answer keys: double triggers removed (truth_clean_1027.py); songs / single-hand
use their labelled hits (hand known), ambiguous / excluded MIDI are not scored.
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P9 = ROOT/'第九次的包_暫定'
sys.path[:0] = [str(P9/'Program'), str(HERE)]
import seventh_core as core
assert Path(core.__file__).resolve().parent == (P9/'Program').resolve()
import exp_1027 as E
import final_1027 as FB
import truth_clean_1027 as TC
L5, H, K, F = FB.L5, FB.H, FB.K, FB.F

OPT = dict(E.DEFAULT, w_rc=0.6, anchor_top=1, conf=0.8, pos_w=0.5, pos_sigma=12.0, pos_all=True, anchor_any=True, anchor_minp=0.7)


def predict(bundle, name):
    gate = bundle['_hit_gate']; P = []
    M = dict(tm={h: np.array([bundle['drum_rotation_means'][f'{h}|{d}'] for d in K.DRUMS]) for h in 'LR'}, _tag='full')
    order = [list(bundle['classes']).index(d) for d in K.DRUMS]   # bundle class order -> research DRUMS order
    E.proba = lambda M_, h, X, w: core.runtime.probabilities(bundle, X, h)[:, order]
    for h in 'LR':
        cs = sorted([c for c in H.CAND[name] if c['hand'] == h], key=lambda c: c['t'])
        if not cs: continue
        ph = gate.inplace_predict(np.stack([H.feats(c) for c in cs]))
        acc = [c for c, q in zip(cs, ph) if q >= bundle['hit_gate']['threshold']]
        if not acc: continue
        X = np.stack([c['x'] for c in acc]).astype(np.float32); t = np.array([c['t'] for c in acc])
        lab, _ = E.run_hand(M, h, X, t, OPT)
        P += [dict(t=float(ti), peak_ms=float(ti), hand=h, drum=K.DRUMS[int(l)]) for ti, l in zip(t, lab)]
    return sorted(P, key=lambda p: (p['t'], p['hand']))


def main():
    rows, names, Lb, T8, R8 = FB.setup()
    bundle, config = core.load_components()
    assert bundle['adaptation']['confidence'] == OPT['conf'] and bundle['position_prior']['sigma_deg'] == OPT['pos_sigma']
    res = {}; cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for name in sorted(H.CAND, key=lambda n: (L5.day_of(n), n)):
        D = L5.day_of(name); P = predict(bundle, name)
        if name in K.TESTS:
            truth = TC.dedupe(L5.TRUTH[name], H.CAND[name])[0]; pairs = L5.match(truth, P); ok = 0
            for i, g in enumerate(truth):
                j = pairs.get(i); good = j is not None and (g['hand'] is None or P[j]['hand'] == g['hand']) and P[j]['drum'] == g['drum']; ok += good
                if g['hand']: c = cells['pretest'][(g['hand'], g['drum'])]; c[0] += good; c[1] += 1
            res[name] = dict(day=D, kind='前測', ok=ok, n=len(truth))
        else:
            used, excl = T8[name] if name in T8 else F.truth_of(name)
            r, cl = F.score(used, excl, P)
            for g, o in cl:
                c = cells['train'][(g['hand'], g['drum'])]; c[0] += o == 'ok'; c[1] += 1
            res[name] = dict(day=D, kind='訓練', ok=r.get('correct', 0), n=r['midi'], wrong_hand=r.get('wrong_hand', 0), missed=r.get('missed', 0), extra=r['extra'])
        print(D, name, res[name], flush=True)
    out = dict(recordings=res, cells={k: {f'{h}|{d}': v for (h, d), v in c.items()} for k, c in cells.items()})
    (HERE/'out/eval_all_9th.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
