"""Should soft / bounce hits (low MIDI velocity) be training data?

Same features (104 venue-free), same per-hand XGBoost+MLP, 2 seeds. Pretests never trained.
Song 14 (moved kit, song 2) is held out as a song test. Variants differ only in song 13 rows:
  V0  no song 13
  V1  song 13 clear hits (labelled L/R, velocity >= 40% of drum median)
  V2  V1 + soft hits you kept (velocity < 40% of drum median)
Scoring vs MIDI: one-to-one time match within 80 ms, then drum. Also accuracy vs coverage when the
model may abstain below a score threshold (high-precision mode).
"""
import json, sys, statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, joblib, warnings
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'第六次的包_暫定2026-10-16/Program'))
import gpu_strict_eval_1020 as G
EXP = ROOT/'第七次的包_暫定2026-10-20/Data/實驗_姿態模型自動標記_20261020'
LAB13 = Path(r'C:/Users/Typelin_Station/Downloads/第七次_13_獨立重核標記_1020_ZH_TW (2).json')
OUT = Path(__file__).resolve().parent/'out'
C = joblib.load(EXP/'實驗快取.joblib')


def rows_from_labels(sname, labels):
    s = C['sessions'][sname]; ev = {str(e['id']): e for e in s['events']}
    med = defaultdict(list)
    for e in ev.values():
        if e.get('name'): med[e['name']].append(e['vel'])
    med = {k: statistics.median(v) for k, v in med.items()}
    edges = []
    for eid, m in labels.items():
        if m['mark'] not in ('L', 'R') or eid not in ev: continue
        e = ev[eid]
        for j, c in enumerate(s['items']):
            p = c['peak']
            if p['hand'] == m['mark'] and abs(p['peak_ms']-e['csv_t']) <= 60: edges.append((abs(p['peak_ms']-e['csv_t']), eid, j))
    own, used = {}, set()
    for d, eid, j in sorted(edges):
        if eid not in own and j not in used: own[eid] = j; used.add(j)
    out, nodet = [], Counter()
    for eid, m in labels.items():
        if m['mark'] not in ('L', 'R') or eid not in ev: continue
        e = ev[eid]; soft = e['vel'] < .4*med[e['name']]
        if eid not in own: nodet['soft' if soft else 'clear'] += 1; continue
        out.append(dict(session=sname, hand=m['mark'], drum=m['drum'], x=np.asarray(s['items'][own[eid]]['x'], np.float32), soft=soft))
    return out, nodet


def train(rows, seed):
    models = {}
    for h in 'LR':
        r = [q for q in rows if q['hand'] == h]; X = np.stack([q['x'] for q in r]); y = [q['drum'] for q in r]
        c = Counter(y); w = np.array([1/np.sqrt(c[v]) for v in y], np.float32); w /= w.mean()
        models[h] = G.fit_models(X, y, w, 'ens', seed=seed)
    return models


def predict(models_list, items):
    out = []
    for h in 'LR':
        it = [c for c in items if c['peak']['hand'] == h]
        if not it: continue
        X = np.stack([np.asarray(c['x'], np.float32) for c in it])
        cls = models_list[0][h][0]; p = np.mean([G.proba(m[h][1], X) for m in models_list], 0)
        for c, q in zip(it, p): out.append(dict(t=c['peak']['peak_ms'], drum=cls[int(q.argmax())], score=float(q.max())))
    return out


def score(truth, preds, thr=0.0):
    keep = [p for p in preds if p['score'] >= thr]
    if not keep: return dict(correct=0, total=len(truth), answered=0, wrong=0)
    tt = np.array([g['t'] for g in truth]); pt = np.array([p['t'] for p in keep])
    dt = np.abs(tt[:, None]-pt[None, :]); cost = np.hstack([np.where(dt <= 80, dt, 1e9), np.full((len(tt), len(tt)), 1e6)])
    r, c = linear_sum_assignment(cost); pairs = {i: j for i, j in zip(r, c) if j < len(pt) and cost[i, j] <= 80}
    det = [(g, keep[pairs[i]] if i in pairs else None) for i, g in enumerate(truth)]
    ok = sum(1 for g, p in det if p and p['drum'] == g['drum']); ans = sum(1 for g, p in det if p)
    soft = [(g, p) for g, p in det if g.get('soft')]; clear = [(g, p) for g, p in det if not g.get('soft')]
    f = lambda L: f"{sum(1 for g,p in L if p and p['drum']==g['drum'])}/{len(L)}"
    return dict(correct=ok, total=len(truth), answered=ans, wrong=ans-ok, rate=round(ok/len(truth), 4),
                precision=round(ok/max(ans, 1), 4), clear=f(clear), soft=f(soft))


def tests():
    T = {}
    for k in ['seventh_01', 'seventh_12', 'seventh_15', 'pretest:round4', 'pretest:round5', 'pretest:round6']:
        t = C['tests'][k]; T[k] = (t['truth'], t['items'])
    s = C['sessions']['seventh_14']; ev = [e for e in s['events'] if e.get('name') and not e.get('out_of_scope') and not e.get('unmapped')]
    med = defaultdict(list)
    for e in ev: med[e['name']].append(e['vel'])
    med = {k: statistics.median(v) for k, v in med.items()}
    T['song14_held_out'] = ([dict(t=e['csv_t'], drum=e['name'], soft=e['vel'] < .4*med[e['name']]) for e in ev], s['items'])
    return T


def main():
    lab = json.loads(LAB13.read_text(encoding='utf-8'))['labels']
    r13, nodet = rows_from_labels('seventh_13', lab)
    print('song13 rows', Counter('soft' if r['soft'] else 'clear' for r in r13), 'no detected peak', dict(nodet))
    base = [dict(session=r['session'], hand=r['hand'], drum=r['drum'], x=np.asarray(r['x'], np.float32)) for r in C['baseline']]
    base += [dict(session=r['session'], hand=r['hand'], drum=r['drum'], x=np.asarray(r['x'], np.float32)) for r in C['auto']
             if r['session'].startswith('seventh_') and r['session'] not in ('seventh_13', 'seventh_14')]
    V = {'V0_無13': base, 'V1_13清晰': base+[r for r in r13 if not r['soft']], 'V2_13清晰+輕擊': base+r13}
    T = tests(); res = {}
    for name, rows in V.items():
        ms = [train(rows, s) for s in range(2)]
        res[name] = {}
        for k, (truth, items) in T.items():
            preds = predict(ms, items)
            res[name][k] = {f'thr{t}': score(truth, preds, t) for t in (0.0, 0.6, 0.8)}
            print(name, k, res[name][k]['thr0.0'], '| thr0.6', {q: res[name][k]['thr0.6'][q] for q in ('rate', 'precision', 'answered')}, flush=True)
    (OUT/'soft_hit_experiment.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
