"""Three versions on every round 4-7 recording + pretests (full pipeline, same scoring as full_report):
  V6  = 10/20 field model (rounds 4-6)
  V7  = new rounds 4-7 committee
  V7R = V7 + causal re-centering (alpha 0.25, after 10 hits of the same hand, past hits only)
Also the 14 pretest cells for each version."""
import json
from collections import Counter, defaultdict
import numpy as np, joblib
import common as K, cells as Q
import full_report as F
import venue_free_runtime_1020 as rt
import causal_hit_detector_1016 as detector
import seventh_core as core
import soft_hit_experiment as S

ROT = slice(83, 104)


def items_of(raw, cfg):
    hands = rt.world.read_raw(raw); prep = rt.world.base.legacy.prepare(hands)
    peaks, _ = detector.detect(hands, cfg); out = []
    for p in sorted(peaks, key=lambda p: p['emitted_ms']):
        s = prep[p['hand']]['rows'][p['available_index']]
        x = rt.feature(prep, p, (s['absolute_yaw']-s['yaw']+180) % 360-180, s['absolute_pitch']-s['pitch'])
        if x is not None: out.append(dict(t=float(p['peak_ms']), hand=p['hand'], x=x))
    return out


def predict(bundle, items, tmean=None, alpha=0.0):
    out = []
    for h in 'LR':
        it = [c for c in items if c['hand'] == h]
        if not it: continue
        X = np.stack([c['x'] for c in it]); Xa = X.copy()
        if alpha:
            for i in range(10, len(it)): Xa[i, ROT] = X[i, ROT]+alpha*(tmean[h]-X[:i, ROT].mean(0))
        p = rt.probabilities(bundle, Xa, h)
        out += [dict(t=c['t'], hand=h, drum=bundle['classes'][int(q.argmax())]) for c, q in zip(it, p)]
    return out


def main():
    _, cfg = core.load_components()
    v7 = joblib.load(K.OUT/'venue_free_committee_rounds4to7_20261021.joblib'); v6 = joblib.load(K.P7/'Model/venue_free_committee_1020.joblib')
    for b in (v6, v7): b['classes'] = [str(c) for c in b['classes']]; rt.attach_boosters(b)
    D = K.build(); tmean = {h: np.mean([r['x'][ROT] for r in D['rows'] if r['hand'] == h], 0) for h in 'LR'}
    VER = {'V6': (v6, 0.0), 'V7': (v7, 0.0), 'V7R': (v7, 0.25)}
    table = []; pre = {v: {} for v in VER}
    for name, title, role in F.ORDER:
        used, excl = F.truth_of(name)
        raw = S.C['tests'][name]['raw'] if name in K.TESTS else S.C['sessions'][name]['raw']
        items = items_of(raw, cfg); row = dict(name=name, title=title, role=role)
        for v, (b, a) in VER.items():
            preds = predict(b, items, tmean, a); res, _ = F.score(used, excl, preds); row[v] = res
            if name in K.TESTS: pre[v][name] = preds
        table.append(row); print(title, {v: f"{row[v]['correct']}/{row[v]['midi']}" for v in VER}, flush=True)
    cells = {v: Q.score_cells(pre[v]) for v in VER}
    for v in VER: print(v, Q.fmt(cells[v]))
    out = dict(table=table, cells={v: dict(acc={f'{a}|{b}': x for (a, b), x in c['acc'].items()}, min=c['min'], mean=c['mean'], overall=c['overall']) for v, c in cells.items()})
    (K.OUT/'compare3.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=int), encoding='utf-8')


if __name__ == '__main__':
    main()
