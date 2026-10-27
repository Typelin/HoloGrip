"""Five-day leave-one-day-out (R4..R8) with the 10/27 data added (8th package pipeline, pretest mode).

R8 = 10/27: single-hand reinforcement (10 takes), four-drum (測試3改) and song (測試2改) labelled by
autolabel_1027.py (uncertain 'review' events excluded); pretest 測試1 is an exam only, never trained.
For each held-out day D the drum committee, per-drum rotation means and hit gate are trained without
any recording of D. Pretests run like the field (snare anchor 5, per-drum offset, top-2 continuity);
songs / single-hand recordings run like lodo_full variant E (no anchor).
Also reports R8 with the shipped 4-day setup (train R4-R7) for comparison. All training on GPU.
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BM = ROOT/'Research/BestModel_20261021'
sys.path[:0] = [str(BM), str(HERE)]
import lodo_anchor as LA
import xgboost as xgb
LF, H, K, F = LA.LF, LA.H, LA.K, LA.F
import hit_cand
from verify_1023 import TRUTH, match

P8 = ROOT/'第八次的包_暫定2026-10-27'
LAB = P8/'Data/自動標記_20261027'
PRE_RAW = next((P8/'Data/模型實測').glob('S20261027*/S20261027_*_raw_100hz_*.csv'))
PRE = 'pretest:round8'
OUT = HERE/'out'; OUT.mkdir(exist_ok=True)
DAYS5 = ['R4', 'R5', 'R6', 'R7', 'R8']


def day_of(name):
    if name.startswith('eighth_') or name == PRE: return 'R8'
    return LF_day_of(name)


LF_day_of = LF.day_of
LF.day_of = day_of


def r8_docs():
    return {f"eighth_{p.name.split('_自動標記')[0]}": json.loads(p.read_text(encoding='utf-8')) for p in sorted(LAB.glob('*_自動標記_20261027.json'))}


def r8_rows(docs):
    """Training rows from autolabel_1027: the labelled event's owned detector item (2.1 g detector, same x as dataset2)."""
    sys.path.insert(0, str(HERE)); import autolabel_1027 as AL
    C = joblib.load(AL.P7/'Data/實驗_姿態模型自動標記_20261020/實驗快取.joblib'); AL.DET = C['detector']
    rows = []
    for name, d in docs.items():
        hands, items, z, _ = AL.A.raw_features(Path(d['raw_csv']), AL.DET)
        ev = {str(e['id']): e for e in d['events']}
        s = dict(events=[dict(id=e['id'], csv_t=e['csv_ms']) for e in d['events']], items=items)
        own = AL.training_rows(s, d['labels'])
        for k, j in own.items():
            m = d['labels'][k]; x = np.asarray(items[j]['x'], np.float32)
            rows.append(dict(session=name, hand=m['mark'], drum=m['drum'], t=items[j]['peak']['peak_ms'], x=x, group='R8', g=float(np.expm1(x[17]))))
        print(name, 'rows', len(own), flush=True)
    return rows


def r8_truth(docs, cand):
    """Used (labelled L/R) and ambiguous event times per R8 recording; pretest truth with hand from glove candidates."""
    T = {}
    for name, d in docs.items():
        used = [dict(t=float(e['csv_ms']), drum=d['labels'][str(e['id'])]['drum'], hand=d['labels'][str(e['id'])]['mark'])
                for e in d['events'] if d['labels'][str(e['id'])]['mark'] in ('L', 'R')]
        amb = [float(e['csv_ms']) for e in d['events'] if d['labels'][str(e['id'])]['mark'] not in ('L', 'R')]
        T[name] = (used, amb)
    import autolabel_1027 as AL
    ev = [e for e in AL.legacy.midi(AL.MID/'測試1.mid')['events'] if e['vel'] >= AL.GHOST_VEL]
    c = cand[PRE]; ct = np.array(sorted(x['t'] for x in c))
    off, rate = AL.align(np.array([e['t'] for e in ev]), ct)
    truth = []
    for e in ev:
        t = e['t']+off; near = [x for x in c if abs(x['t']-t) <= 60]
        hand = max(near, key=lambda x: x['g'])['hand'] if near else None
        truth.append(dict(id=e['id'], t=t, drum=e['name'], hand=hand))
    print('pretest R8 offset', round(off), 'align', round(rate, 3), 'hands', Counter(g['hand'] for g in truth), flush=True)
    return T, truth


def gate_labels(name, T8):
    if name in T8:
        used, amb_t = T8[name]; cands = H.CAND[name]
        lab = np.full(len(cands), -1); ut = np.array([u['t'] for u in used])
        allt = np.r_[ut, np.array(amb_t, float)]; lo, hi = allt.min()-200, allt.max()+200
        from scipy.optimize import linear_sum_assignment
        for h in 'LR':
            ci = [i for i, c in enumerate(cands) if c['hand'] == h]; ui = [j for j, u in enumerate(used) if u['hand'] == h]
            if ci and ui:
                dt = np.abs(np.array([cands[i]['t'] for i in ci])[:, None]-ut[ui][None, :]); cost = np.where(dt <= 60, dt, 1e9)
                r, cc = linear_sum_assignment(cost)
                for a, b in zip(r, cc):
                    if cost[a, b] <= 60: lab[ci[a]] = 1
        amb = np.array(amb_t) if len(amb_t) else np.array([1e18])
        for i, c in enumerate(cands):
            if lab[i] == 1: continue
            lab[i] = 2 if (np.min(np.abs(amb-c['t'])) <= 80 or not lo <= c['t'] <= hi) else 0
        return lab, len(used)
    return H.labels(name)


def train_day(rows, names, L, hold):
    tr = [r for r in rows if day_of(r['session']) not in hold]
    com = K.train(tr, {}, seeds=(0, 1))
    tm = {h: np.array([np.mean([r['x'][83:104] for r in tr if r['hand'] == h and r['drum'] == d], 0) if any(r['hand'] == h and r['drum'] == d for r in tr) else np.zeros(21)
                       for d in K.DRUMS]) for h in 'LR'}
    gn = [n for n in names if n not in K.TESTS and day_of(n) not in hold]
    Xg = np.vstack([np.stack([H.feats(c) for c in H.CAND[n]])[L[n][0] != 2] for n in gn]); yg = np.concatenate([L[n][0][L[n][0] != 2] for n in gn])
    gate = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, device='cuda', tree_method='hist', random_state=0).fit(Xg, yg)
    return com, tm, gate, len(tr)


def exam(D, names, com, tm, gate, T8):
    cells = defaultdict(Counter); pre = Counter(); song = Counter()
    for n in [x for x in names if day_of(x) == D]:
        if n in K.TESTS:
            P = LA.run(n, com, tm, gate, 5, 5); truth = TRUTH[n]; pairs = match(truth, P)
            for i, g in enumerate(truth):
                j = pairs.get(i); ok = j is not None and (g['hand'] is None or P[j]['hand'] == g['hand']) and P[j]['drum'] == g['drum']
                pre['ok'] += ok; pre['n'] += 1
                if g['hand']: cells[(g['hand'], g['drum'])]['ok'] += ok; cells[(g['hand'], g['drum'])]['n'] += 1
        else:
            P = LF.run_recording(n, com, tm, gate, 'E')
            used, excl = T8[n] if n in T8 else F.truth_of(n)
            res, _ = F.score(used, excl, P)
            song['ok'] += res.get('correct', 0); song['n'] += res['midi']; song['extra'] += res['extra']; song['matched'] += res['matched']
    acc = {f'{h}{d}': round(v['ok']/v['n'], 3) for (h, d), v in cells.items()}
    return dict(pretest=round(pre['ok']/max(pre['n'], 1), 4), pretest_n=pre['n'], cells=acc,
                cell_mean=round(float(np.mean(list(acc.values()))), 4) if acc else None,
                songs=round(song['ok']/max(song['n'], 1), 4), songs_n=song['n'], hit_recall=round(song['matched']/max(song['n'], 1), 4), extra=song['extra'])


def main():
    docs = r8_docs()
    cf = OUT/'r8_cache.joblib'
    if cf.exists():
        R8 = joblib.load(cf)
    else:
        jobs = [(PRE, str(PRE_RAW), 1.5, {})]+[(n, d['raw_csv'], 1.5, {}) for n, d in docs.items()]
        with Pool(6) as pool: cand = dict(pool.imap_unordered(hit_cand.work, jobs))
        R8 = dict(cand=cand, rows=r8_rows(docs)); joblib.dump(R8, cf)
    H.CAND.update(R8['cand'])
    T8, truth = r8_truth(docs, R8['cand']); TRUTH[PRE] = truth
    if PRE not in K.TESTS: K.TESTS.append(PRE)
    rows = joblib.load(K.OUT/'dataset2.joblib')['rows']+R8['rows']
    print('training rows', len(rows), 'R8', len(R8['rows']), Counter((r['hand'], r['drum']) for r in R8['rows']), flush=True)
    names = list(H.CAND); L = {n: gate_labels(n, T8) for n in names}
    out = {}
    # shipped setup (4 days) on the new day
    com, tm, gate, ntr = train_day(rows, names, L, {'R8'})
    out['R8_trained_R4toR7'] = r = exam('R8', names, com, tm, gate, T8); r['train_rows'] = ntr
    print('R8 用 R4~R7 訓練（現行包的做法）', json.dumps(r, ensure_ascii=False), flush=True)
    for D in DAYS5[:-1]:
        com, tm, gate, ntr = train_day(rows, names, L, {D})
        out[f'{D}_held_5day'] = r = exam(D, names, com, tm, gate, T8); r['train_rows'] = ntr
        print(D, '留出（其他四天含 R8 訓練）', json.dumps(r, ensure_ascii=False), flush=True)
        (OUT/'lodo5.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT/'lodo5.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
