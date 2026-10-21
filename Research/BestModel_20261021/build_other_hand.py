"""Add other-hand context per hit (causal: other hand's samples up to hit + 80 ms):
other hand's zero-relative pose at the hit (rotvec + yaw/pitch), its activity around the hit,
and the angle between the two hands' zero-relative directions. Writes out/dataset3.joblib."""
import numpy as np, joblib
from collections import defaultdict
from scipy.spatial.transform import Rotation as Rot
import common as K
import heading_free_model_1019 as H
import soft_hit_experiment as S


def rel_at(prep, hand, t):
    rows = prep[hand]['rows']; times = prep[hand]['times']
    if not rows: return None
    avail = int(np.searchsorted(times, t+80, 'right'))-1
    if avail < 0: return None
    s = rows[avail]; zy = (s['absolute_yaw']-s['yaw']+180) % 360-180; zp = s['absolute_pitch']-s['pitch']
    j = int(np.argmin(np.abs(times[:avail+1]-t))); r = rows[j]
    A = Rot.from_euler('ZYX', [zy, zp, 0.], degrees=True); Hd = Rot.from_euler('Z', -zy, degrees=True)
    rel = Hd*Rot.from_euler('ZYX', [r['absolute_yaw'], r['absolute_pitch'], r['roll']], degrees=True)*A.inv()*Hd.inv()
    lo = int(np.searchsorted(times, t-150)); act = [abs(q['mag']-1) for q in rows[lo:avail+1]]
    return rel, (max(act) if act else 0.)


def other_feats(prep, hand, t):
    oh = 'R' if hand == 'L' else 'L'; me = rel_at(prep, hand, t); ot = rel_at(prep, oh, t)
    if me is None or ot is None: return np.full(10, np.nan, np.float32)
    e = ot[0].as_euler('ZYX', degrees=True)
    v1 = me[0].apply([1, 0, 0]); v2 = ot[0].apply([1, 0, 0])
    ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
    return np.r_[ot[0].as_rotvec()/np.pi, np.sin(np.radians(e[0])), np.cos(np.radians(e[0])), e[1]/90, e[2]/180, min(ot[1], 8)/8, ang/180, (v1-v2)[2]].astype(np.float32)[:10]


def main():
    D = joblib.load(K.OUT/'dataset2.joblib'); C = S.C
    by = defaultdict(list)
    for i, r in enumerate(D['rows']): by[r['session']].append(i)
    for s, idx in by.items():
        prep = H.world.base.legacy.prepare(H.world.read_raw(C['sessions'][s]['raw']))
        for i in idx: D['rows'][i]['other'] = other_feats(prep, D['rows'][i]['hand'], D['rows'][i]['t'])
        print(s, flush=True)
    for k, t in D['tests'].items():
        prep = H.world.base.legacy.prepare(H.world.read_raw(C['tests'][k]['raw']))
        for c in t['items']: c['other'] = other_feats(prep, c['hand'], c['t'])
        print(k, flush=True)
    joblib.dump(D, K.OUT/'dataset3.joblib'); print('done')


if __name__ == '__main__':
    main()
