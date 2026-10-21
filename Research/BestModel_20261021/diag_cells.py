"""Diagnostics: cross-day (group-out) recall per hand x drum, and pretest confusion, with base_ens."""
import json
from collections import Counter, defaultdict
import numpy as np, joblib
import common as K


def main():
    D = joblib.load(K.OUT/'dataset2.joblib'); rows = D['rows']
    cell = defaultdict(lambda: [0, 0]); conf = Counter(); train_n = Counter((r['hand'], r['drum'], r['group']) for r in rows)
    for g in ['R4', 'R5', 'R6', 'R7o', 'R7m']:
        com = K.train([r for r in rows if r['group'] != g], {}, seeds=(0,))
        for h in 'LR':
            te = [r for r in rows if r['group'] == g and r['hand'] == h]
            p = K.proba(com, h, np.stack([r['x'] for r in te]))
            for r, q in zip(te, p):
                pr = K.DRUMS[int(q.argmax())]; cell[(h, r['drum'], g)][1] += 1; cell[(h, r['drum'], g)][0] += pr == r['drum']
                if pr != r['drum']: conf[(h, r['drum'], pr)] += 1
    out = []
    for h in 'LR':
        for d in K.DRUMS:
            line = {g: (f"{cell[(h, d, g)][0]}/{cell[(h, d, g)][1]}" if cell[(h, d, g)][1] else '-') for g in ['R4', 'R5', 'R6', 'R7o', 'R7m']}
            tot = sum(cell[(h, d, g)][1] for g in ['R4', 'R5', 'R6', 'R7o', 'R7m']); ok = sum(cell[(h, d, g)][0] for g in ['R4', 'R5', 'R6', 'R7o', 'R7m'])
            out.append(dict(hand=h, drum=d, recall=round(ok/max(tot, 1), 3), n=tot, by_group=line)); print(h, d, round(ok/max(tot, 1), 3), tot, line, flush=True)
    print('top confusions', conf.most_common(15))
    (K.OUT/'diag_cells.json').write_text(json.dumps(dict(cells=out, confusions=[[*k, v] for k, v in conf.most_common(30)]), ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
