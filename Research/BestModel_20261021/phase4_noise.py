"""Phase 4: label-noise check. Out-of-fold (5-fold stratified within each recording) predictions for
every training hit; hits the model confidently assigns to a different drum are suspects.
Experiment: drop suspects (prob of labelled drum < p) and retrain; compare pretests.
Writes out/phase4_suspects.csv for human spot-checks."""
import csv, json, time
import numpy as np, joblib
import common as K


def main():
    D = joblib.load(K.OUT/'dataset2.joblib'); rows = D['rows']; n = len(rows)
    rng = np.random.default_rng(0); fold = rng.integers(0, 5, n)
    p_lab = np.zeros(n); pred = [''] * n; pmax = np.zeros(n)
    for f in range(5):
        tr = [r for r, k in zip(rows, fold) if k != f]; com = K.train(tr, {}, seeds=(0,))
        for h in 'LR':
            idx = [i for i in range(n) if fold[i] == f and rows[i]['hand'] == h]
            p = K.proba(com, h, np.stack([rows[i]['x'] for i in idx]))
            for i, q in zip(idx, p):
                p_lab[i] = q[K.DRUMS.index(rows[i]['drum'])]; pred[i] = K.DRUMS[int(q.argmax())]; pmax[i] = q.max()
        print('fold', f, flush=True)
    sus = [i for i in range(n) if pred[i] != rows[i]['drum'] and p_lab[i] < .05 and pmax[i] > .9]
    with (K.OUT/'phase4_suspects.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh); w.writerow(['session', 'hand', 't_ms', 'label', 'model', 'p_label', 'p_model', 'peak_g'])
        for i in sorted(sus, key=lambda i: p_lab[i]):
            r = rows[i]; w.writerow([r['session'], r['hand'], round(r['t']), r['drum'], pred[i], round(p_lab[i], 4), round(pmax[i], 3), round(r['g'], 2)])
    from collections import Counter
    print('oof acc', round(float(np.mean([pred[i] == rows[i]['drum'] for i in range(n)])), 4), 'suspects', len(sus), Counter(rows[i]['session'] for i in sus).most_common(12), flush=True)
    res = {}
    for thr in (0.0, 0.01, 0.05):
        keep = [r for r, pl, pr, pm in zip(rows, p_lab, pred, pmax) if not (pr != r['drum'] and pl < thr and pm > .9)] if thr else rows
        com = K.train(keep, {}, seeds=(0, 1)); _, summ = K.exam(com, D, thrs=(0.0, 0.6))
        res[str(thr)] = dict(kept=len(keep), pretest=summ); print('drop<', thr, len(keep), summ, flush=True)
    (K.OUT/'phase4.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
