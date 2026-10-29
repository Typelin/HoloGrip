"""Hit-by-hit trace of selected cells with the installed 9th package model (batch = field logic)."""
import sys
import numpy as np
sys.argv = ['x']
import eval_all_9th as A
E, K, L5, H = A.E, A.K, A.L5, A.H
rows, names, Lb, T8, R8 = A.FB.setup(); bundle, _ = A.core.load_components()
order = [list(bundle['classes']).index(d) for d in K.DRUMS]
E.proba = lambda M_, h, X, w: A.core.runtime.probabilities(bundle, X, h)[:, order]
M = dict(tm={h: np.array([bundle['drum_rotation_means'][f'{h}|{d}'] for d in K.DRUMS]) for h in 'LR'}, _tag='full')
tmY = {h: L5.LF.yp(np.hstack([np.zeros((7, 83), np.float32), M['tm'][h].astype(np.float32)])) for h in 'LR'}


def trace(nm, h, drums, show_ok=False):
    cs = sorted([c for c in H.CAND[nm] if c['hand'] == h], key=lambda c: c['t'])
    ph = bundle['_hit_gate'].inplace_predict(np.stack([H.feats(c) for c in cs])); acc = [c for c, q in zip(cs, ph) if q >= .4]
    X = np.stack([c['x'] for c in acc]).astype(np.float32); t = np.array([c['t'] for c in acc])
    lab, anc = E.run_hand(M, h, X, t, A.OPT); D = E.LAST_DEBUG; p = D['p']; Y = D['Y']
    print(f'== {nm} {h} 錨點={anc} 訓練位置', {d[:2]: tmY[h][i].round(0).tolist() for i, d in enumerate(K.DRUMS)})
    T = A.TC.dedupe(L5.TRUTH[nm], H.CAND[nm])[0]
    for g in T:
        if g['hand'] != h or g['drum'] not in drums: continue
        j = int(np.argmin(np.abs(t-g['t'])))
        if abs(t[j]-g['t']) > 80: print('  miss'); continue
        ok = K.DRUMS[lab[j]] == g['drum']
        if ok and not show_ok: continue
        top = np.argsort(-p[j])[:3]
        print(f"  #{j:3d} 真={g['drum'][:2]} 判={K.DRUMS[lab[j]][:2]} {'✓' if ok else '✗'} 前三={'/'.join(f'{K.DRUMS[k][:2]}{p[j][k]:.2f}' for k in top)} 位置={Y[j].round(0).tolist()}")


if __name__ == '__main__':
    for nm, h, d in [('seventh_12', 'R', ['落地 Tom']), ('seventh_15', 'R', ['Ride']), ('seventh_01', 'R', ['中音 Tom']), ('seventh_12', 'L', ['小鼓'])]:
        trace(nm, h, d, show_ok=True)
