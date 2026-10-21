"""Tune the prototype-aligned per-take offset and check safety on the held-out song (04, in time order)
and on single-drum takes (round-7 single-hand sessions, here as a harm check)."""
import json, itertools
import numpy as np
import common as K, cells as Q, h_common as H
from h2_style_offset import train_w

ROT = H.ROT


def make_post(tm, alpha, conf, minh, passes):
    def post(com, h, it, X):
        p = K.proba(com, h, X); Xa = X
        for _ in range(passes):
            pred = p.argmax(1); Xn = X.copy()
            for i in range(len(X)):
                past = [j for j in range(i) if p[j].max() >= conf]
                if len(past) < minh: continue
                gaps, ws = [], []
                for d in set(pred[past]):
                    js = [j for j in past if pred[j] == d]; key = (h, K.DRUMS[d])
                    if key in tm: gaps.append(tm[key]-X[js, ROT].mean(0)); ws.append(len(js))
                if gaps: Xn[i, ROT] = X[i, ROT]+alpha*np.average(np.array(gaps), axis=0, weights=ws)
            Xa = Xn; p = K.proba(com, h, Xa)
        return p
    return post


def seq_acc(com, rows, post):
    ok = n = 0
    for s in sorted({r['session'] for r in rows}):
        for h in 'LR':
            t = sorted([r for r in rows if r['session'] == s and r['hand'] == h], key=lambda r: r['t'])
            if not t: continue
            X = np.stack([r['x'] for r in t]); p = post(com, h, t, X) if post else K.proba(com, h, X)
            ok += sum(K.DRUMS[int(q.argmax())] == r['drum'] for q, r in zip(p, t)); n += len(t)
    return ok/n


def main():
    train, hold, tests = H.load(with_timing=False)
    com = train_w(train, lambda r: r['x'], lambda r: 1.)
    tm = {(h, d): np.mean([r['x'][ROT] for r in train if r['hand'] == h and r['drum'] == d], 0) for h in 'LR' for d in K.DRUMS
          if any(r['hand'] == h and r['drum'] == d for r in train)}
    singles = [r for r in train if r['session'] in ('seventh_06', 'seventh_07', 'seventh_08', 'seventh_09', 'seventh_11', 'sixth_S06')]
    out = {}
    print('no post', Q.fmt(H.pretest(com, tests)), 'song04', round(seq_acc(com, hold, None), 4), 'singles', round(seq_acc(com, singles, None), 4), flush=True)
    for alpha, conf, minh, passes in itertools.product([0.75, 1.0, 1.25], [0.5, 0.7, 0.9], [5, 10, 20], [1, 2]):
        post = make_post(tm, alpha, conf, minh, passes)
        r = H.pretest(com, tests, post=post); s = seq_acc(com, hold, post); g = seq_acc(com, singles, post)
        key = f'a{alpha}_c{conf}_m{minh}_p{passes}'
        out[key] = dict(min=r['min'], mean=r['mean'], song=s, singles=g, acc={f'{a}{b}': v for (a, b), v in r['acc'].items()})
        print(key, Q.fmt(r), 'song04', round(s, 4), 'singles', round(g, 4), flush=True)
        (K.OUT/'h3.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
