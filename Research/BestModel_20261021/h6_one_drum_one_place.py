"""'One drum, one place' (code logic, causal). Per take and hand, hits are clustered online by their
zero-relative yaw/pitch at impact. Each cluster pools the log-probabilities of its members. Clusters
with >= m members are matched one-to-one to drums (Hungarian on pooled scores), so two separate places
cannot both be the same drum. The current hit takes its cluster's drum (if the cluster is big enough),
otherwise the model's own answer. Only hits up to the current one are used."""
import itertools
import numpy as np, joblib
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as Rot
import common as K, cells as Q, h_common as H
import full_report as F
from h3_proto_grid import make_post, seq_acc
from h5_cymbal_rescue import items_at
import soft_hit_experiment as S
import seventh_core as core


def yp_of(X):
    v = X[:, 83:104].reshape(len(X), 7, 3)[:, 4]*np.pi
    return Rot.from_rotvec(v).as_euler('ZYX', degrees=True)[:, :2]


def odop(P, YP, radius, m, sep):
    """P: (n,7) probs in time order; YP: (n,2) yaw/pitch. Returns labels (n,)."""
    n = len(P); lab = P.argmax(1).copy(); cent, members = [], []
    logp = np.log(P+1e-6)
    for i in range(n):
        if cent:
            d = np.array([np.hypot((YP[i, 0]-c[0]+180) % 360-180, YP[i, 1]-c[1]) for c in cent]); j = int(d.argmin())
        if not cent or d[j] > radius: cent.append(YP[i].copy()); members.append([i]); j = len(cent)-1
        else: members[j].append(i); cent[j] = YP[members[j]].mean(0)
        big = [k for k in range(len(cent)) if len(members[k]) >= m]
        if j not in big: continue
        score = np.array([logp[members[k]].sum(0) for k in big])
        # only enforce uniqueness among clusters that are clearly separated
        r, c = linear_sum_assignment(-score)
        assign = dict(zip([big[x] for x in r], c))
        own = int(score[big.index(j)].argmax())
        # keep own best unless another separated cluster has a stronger claim on the same drum
        rivals = [k for k in big if k != j and int(score[big.index(k)].argmax()) == own and
                  np.hypot((cent[k][0]-cent[j][0]+180) % 360-180, cent[k][1]-cent[j][1]) > sep]
        if rivals and any(score[big.index(k)][own] > score[big.index(j)][own]*len(members[j])/len(members[k]) for k in rivals):
            lab[i] = assign.get(j, own)
        else: lab[i] = own
    return lab


def main():
    com = joblib.load(K.OUT/'h_best_com.joblib'); train, hold, tests = H.load(with_timing=False); _, cfg = core.load_components()
    tm = {(h, d): np.mean([r['x'][H.ROT] for r in train if r['hand'] == h and r['drum'] == d], 0) for h in 'LR' for d in K.DRUMS
          if any(r['hand'] == h and r['drum'] == d for r in train)}
    post = make_post(tm, 1.0, 0.7, 20, 2)
    raws = {k: S.C['tests'][k]['raw'] for k in K.TESTS}; raws['song04'] = S.C['sessions']['seventh_04']['raw']
    for s in ('seventh_06', 'seventh_09', 'sixth_S06', 'sixth_S07'): raws[s] = S.C['sessions'][s]['raw']
    its = {k: items_at(r, cfg, 1.7) for k, r in raws.items()}
    probs = {}
    for k, items in its.items():
        probs[k] = {}
        for h in 'LR':
            it = sorted([c for c in items if c['hand'] == h], key=lambda c: c['t'])
            if not it: continue
            X = np.stack([c['x'] for c in it]); probs[k][h] = (it, post(com, h, it, X), yp_of(X))
    used, excl = F.truth_of('seventh_04')
    for radius, m, sep in itertools.product([8, 12, 16], [5, 10], [15, 25]):
        out = {}
        for k in probs:
            P = []
            for h, (it, p, yp) in probs[k].items():
                lab = odop(p, yp, radius, m, sep)
                for c, q, l in zip(it, p, lab):
                    d = K.DRUMS[int(l)]
                    if c['g'] >= 2.1 or (d in ('Hi-Hat', 'Crash', 'Ride') and q.max() >= .6): P.append(dict(t=c['t'], hand=h, drum=d))
            out[k] = P
        r = Q.score_cells({k: out[k] for k in K.TESTS}); res, _ = F.score(used, excl, out['song04'])
        singles = []
        for s in ('seventh_06', 'seventh_09', 'sixth_S06', 'sixth_S07'):
            u, e = F.truth_of(s); rr, _ = F.score(u, e, out[s]); singles.append(f"{rr.get('correct',0)}/{rr['midi']}")
        print(f'r{radius} m{m} sep{sep}', Q.fmt(r), f"| song04 {res.get('correct',0)}/{res['midi']} | singles {' '.join(singles)}", flush=True)


if __name__ == '__main__':
    main()
