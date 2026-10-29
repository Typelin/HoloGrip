"""Per-take pretest cells with the installed 9th package model under option overrides; also 40-recording total."""
import sys, json
import numpy as np
sys.argv = ['x']
import eval_all_9th as A
rows, names, Lb, T8, R8 = A.FB.setup(); bundle, _ = A.core.load_components()
BASE = dict(A.OPT); PRE = [n for v in A.FB.PRETESTS.values() for n in v]


def run(over, songs=False):
    A.OPT.clear(); A.OPT.update(dict(BASE, **over)); tot = [0, 0]; below = []; ncell = 0
    for nm in PRE:
        P = A.predict(bundle, nm); T = A.TC.dedupe(A.L5.TRUTH[nm], A.H.CAND[nm])[0]; pr = A.L5.match(T, P); cells = {}
        for i, g in enumerate(T):
            j = pr.get(i); ok = j is not None and (g['hand'] is None or P[j]['hand'] == g['hand']) and P[j]['drum'] == g['drum']
            tot[0] += ok; tot[1] += 1
            if g['hand']: c = cells.setdefault(g['hand']+g['drum'], [0, 0]); c[0] += ok; c[1] += 1
        ncell += len(cells); below += [f'{nm[-2:]}{k}:{a}/{b}' for k, (a, b) in cells.items() if a/b < .85]
    out = dict(pre=tot, below=below, ncell=ncell)
    if songs:
        s = [0, 0]
        for nm in A.H.CAND:
            if nm in A.K.TESTS: continue
            used, excl = T8[nm] if nm in T8 else A.F.truth_of(nm); r, _ = A.F.score(used, excl, A.predict(bundle, nm)); s[0] += r.get('correct', 0); s[1] += r['midi']
        out['train'] = s
    return out


if __name__ == '__main__':
    for label, over in json.loads(sys.argv[1] if len(sys.argv) > 1 else '[]') or [('現況', {})]:
        r = run(over); print(label, '前測', r['pre'], f"{r['pre'][0]/r['pre'][1]*100:.1f}%", '未達格', len(r['below']), '/', r['ncell'], r['below'], flush=True)
