"""Goal step 1: strong drums absorb neighbours -> rebalance.
(a) class weighting: sqrt vs full inverse frequency; (b) post-hoc prior correction p/prior^tau per hand.
Saves raw probabilities so later steps can reuse them."""
import json
from collections import Counter
import numpy as np, joblib
import common as K, cells as Q


def main():
    D = K.build(); rows = D['rows']; res = {}
    for bal in ['sqrt', 'full']:
        com = K.train(rows, dict(class_bal=bal), seeds=(0, 1, 2))
        probs = {k: {h: K.proba(com, h, np.stack([c['x'] for c in t['items'] if c['hand'] == h])) for h in 'LR'} for k, t in D['tests'].items()}
        joblib.dump(probs, K.OUT/f'g1_probs_{bal}.joblib')
        prior = {h: np.array([max(sum(1 for r in rows if r['hand'] == h and r['drum'] == d), 1) for d in K.DRUMS], float) for h in 'LR'}
        for h in 'LR': prior[h] /= prior[h].sum()
        for tau in [0, .25, .5, .75, 1.0]:
            def fn(h, it, k=None):
                return None
            preds = {}
            for k, t in D['tests'].items():
                P = []
                for h in 'LR':
                    it = [c for c in t['items'] if c['hand'] == h]; p = probs[k][h]/prior[h]**tau
                    P += [dict(t=c['t'], hand=h, drum=K.DRUMS[int(q.argmax())]) for c, q in zip(it, p)]
                preds[k] = P
            r = Q.score_cells(preds); res[f'{bal}_tau{tau}'] = r
            print(bal, tau, Q.fmt(r), flush=True)
    joblib.dump(res, K.OUT/'g1_results.joblib')


if __name__ == '__main__':
    main()
