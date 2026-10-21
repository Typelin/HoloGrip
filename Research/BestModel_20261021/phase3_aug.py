"""Phase 3: drumhead-zero jitter augmentation (train) and zero-jitter averaging (test, TTA).

Only the 21 zero-relative rotation dims change; trajectory summary is heading-free and unaffected.
TTA averages predictions over a small fixed grid of zero offsets for each hit independently
(still causal: uses only that hit's window and the recorded zero).
"""
import json, time
import numpy as np, joblib
import common as K
import venue_free_runtime_1020 as rt

IDX7 = [4, 6, 8, 9, 10, 11, 12]  # dense DTS (-200..80 step 20) positions of -120,-80,-40,-20,0,20,40


def x_with_zero(r, dy, dp):
    pose7 = np.asarray(r['pose'])[IDX7]
    return np.r_[r['x'][:83], rt.zero_relative_rotation(pose7, r['zero'][0]+dy, r['zero'][1]+dp)].astype(np.float32)


def augment(rows, a, copies, seed=0):
    if a == 0: return rows
    rng = np.random.default_rng(seed); out = list(rows)
    for r in rows:
        for _ in range(copies):
            out.append(dict(r, x=x_with_zero(r, rng.uniform(-a, a), rng.uniform(-a/2, a/2))))
    return out


def tta_predict(com, items, a):
    grid = [(0, 0)] if a == 0 else [(0, 0), (a, 0), (-a, 0), (0, a/2), (0, -a/2)]
    out = []
    for h in 'LR':
        it = [c for c in items if c['hand'] == h]
        p = np.mean([K.proba(com, h, np.stack([x_with_zero(c, dy, dp) for c in it])) for dy, dp in grid], 0)
        out += [dict(t=c['t'], g=c['g'], drum=K.DRUMS[int(q.argmax())], score=float(q.max())) for c, q in zip(it, p)]
    return out


def evaluate(com, D, tta):
    res = {}
    for k, t in D['tests'].items():
        preds = tta_predict(com, t['items'], tta)
        res[k] = {str(th): K.score_pretest(t['truth'], preds, th) for th in (0.0, 0.6)}
    summ = {}
    for th in ('0.0', '0.6'):
        rates = [res[k][th]['correct']/res[k][th]['total'] for k in res]
        summ[th] = dict(mean_rate=round(float(np.mean(rates)), 4), min_rate=round(float(np.min(rates)), 4),
                        precision=round(sum(res[k][th]['correct'] for k in res)/max(sum(res[k][th]['out'] for k in res), 1), 4))
    return res, summ


def cv(rows, a, copies, tta):
    out = {}
    for g in ['R4', 'R5', 'R6', 'R7o', 'R7m']:
        tr = augment([r for r in rows if r['group'] != g], a, copies); te = [r for r in rows if r['group'] == g]
        com = K.train(tr, {}, seeds=(0,)); ok = n = 0
        for h in 'LR':
            t = [r for r in te if r['hand'] == h]
            grid = [(0, 0)] if tta == 0 else [(0, 0), (tta, 0), (-tta, 0), (0, tta/2), (0, -tta/2)]
            p = np.mean([K.proba(com, h, np.stack([x_with_zero(r, dy, dp) for r in t])) for dy, dp in grid], 0)
            ok += sum(K.DRUMS[int(q.argmax())] == r['drum'] for q, r in zip(p, t)); n += len(t)
        out[g] = round(ok/n, 4)
    out['mean'] = round(float(np.mean([out[g] for g in ['R4', 'R5', 'R6', 'R7o', 'R7m']])), 4); return out


def main():
    D = joblib.load(K.OUT/'dataset2.joblib'); rows = D['rows']
    chk = max(float(np.abs(x_with_zero(r, 0, 0)-r['x']).max()) for r in rows[:300]+rows[-300:])
    print('rot recompute max diff', chk, flush=True)
    f = K.OUT/'phase3.jsonl'; done = set()
    if f.exists(): done = {json.loads(l)['name'] for l in f.read_text(encoding='utf-8').splitlines() if l.strip()}
    for a, copies, tta in [(0, 0, 0), (0, 0, 10), (10, 3, 0), (10, 3, 10), (15, 3, 0), (15, 3, 15), (20, 3, 15), (10, 6, 10)]:
        name = f'aug{a}x{copies}_tta{tta}'
        if name in done: continue
        t0 = time.time(); com = K.train(augment(rows, a, copies), {}, seeds=(0, 1)); res, summ = evaluate(com, D, tta); c = cv(rows, a, copies, tta)
        rec = dict(name=name, pretest=summ, per={k: round(v['0.0']['correct']/v['0.0']['total'], 4) for k, v in res.items()}, cv=c, sec=round(time.time()-t0))
        with f.open('a', encoding='utf-8') as fh: fh.write(json.dumps(rec, ensure_ascii=False)+'\n')
        print(name, summ['0.0'], 'p0.6', summ['0.6'], 'cv', c, rec['per'], rec['sec'], flush=True)


if __name__ == '__main__':
    main()
