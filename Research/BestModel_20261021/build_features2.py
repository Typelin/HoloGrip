"""Phase 2 features from Raw (causal, per hit): full heading-free vector and a dense zero-relative
rotation path. Same rows as dataset.joblib plus the peak time; pretest items from detector peaks.
Writes out/dataset2.joblib. CPU only (feature extraction), training stays on GPU.
"""
import json, sys
from collections import defaultdict
import numpy as np, joblib
from scipy.spatial.transform import Rotation as Rot
import common as K
import heading_free_model_1019 as H
import soft_hit_experiment as S

DTS = np.arange(-200, 81, 20)


def prep_session(raw):
    hands = H.world.read_raw(raw); return H.world.base.legacy.prepare(hands)


def dense_rot(prep, hand, t, avail):
    rows = prep[hand]['rows']; times = prep[hand]['times'][:avail+1]
    s = rows[avail]; zy = (s['absolute_yaw']-s['yaw']+180) % 360-180; zp = s['absolute_pitch']-s['pitch']
    pose = []
    for dt in DTS:
        r = rows[int(np.argmin(np.abs(times-(t+dt))))]; pose.append([r['absolute_yaw'], r['absolute_pitch'], r['roll']])
    A = Rot.from_euler('ZYX', [zy, zp, 0.], degrees=True); Hd = Rot.from_euler('Z', -zy, degrees=True)
    rel = Hd*Rot.from_euler('ZYX', np.array(pose), degrees=True)*A.inv()*Hd.inv()
    eul = rel.as_euler('ZYX', degrees=True)
    return np.r_[rel.as_rotvec().ravel()/np.pi, np.sin(np.radians(eul[:, 0])), np.cos(np.radians(eul[:, 0])), eul[:, 1]/90, eul[:, 2]/180].astype(np.float32), np.array(pose)


def feats(prep, hand, t):
    times = prep[hand]['times']; avail = int(np.searchsorted(times, t+80, 'right'))-1
    peak = dict(hand=hand, peak_ms=t, available_index=avail)
    hf = H.heading_free(prep, peak, False)
    if hf is None: return None
    d, pose = dense_rot(prep, hand, t, avail)
    s = prep[hand]["rows"][avail]
    return hf.astype(np.float32), d, pose, ((s["absolute_yaw"]-s["yaw"]+180) % 360-180, s["absolute_pitch"]-s["pitch"])


def main():
    C = S.C; rows = []
    for r in C['baseline']: rows.append(dict(session=r['session'], hand=r['hand'], drum=r['drum'], t=r['t'], x=np.asarray(r['x'], np.float32)))
    for code in ['03', '04', '05', '13', '14']:
        L = json.loads((K.LAB/f'{code}_完成標記_20261021.json').read_text(encoding='utf-8'))['labels']
        s = C['sessions'][f'seventh_{code}']; ev = {str(e['id']): e for e in s['events']}
        edges = []
        for eid, m in L.items():
            if m['mark'] not in ('L', 'R') or eid not in ev: continue
            for j, c in enumerate(s['items']):
                p = c['peak']
                if p['hand'] == m['mark'] and abs(p['peak_ms']-ev[eid]['csv_t']) <= 60: edges.append((abs(p['peak_ms']-ev[eid]['csv_t']), eid, j))
        own, used = {}, set()
        for _, eid, j in sorted(edges):
            if eid not in own and j not in used: own[eid] = j; used.add(j)
        for eid, j in own.items():
            c = s['items'][j]; rows.append(dict(session=f'seventh_{code}', hand=L[eid]['mark'], drum=L[eid]['drum'], t=c['peak']['peak_ms'], x=np.asarray(c['x'], np.float32)))
    for r in C['auto']:
        if r['session'] in [f'seventh_{i:02d}' for i in range(6, 12)]:
            rows.append(dict(session=r['session'], hand=r['hand'], drum=r['drum'], t=r['t'], x=np.asarray(r['x'], np.float32)))
    by = defaultdict(list)
    for i, r in enumerate(rows): by[r['session']].append(i)
    keep = []
    for sname, idx in by.items():
        prep = prep_session(C['sessions'][sname]['raw'])
        for i in idx:
            f = feats(prep, rows[i]['hand'], rows[i]['t'])
            if f is None: continue
            rows[i]['hf'], rows[i]['dense'], rows[i]['pose'], rows[i]['zero'] = f; rows[i]['group'] = K.group_of(sname); rows[i]['g'] = float(np.expm1(rows[i]['x'][17])); keep.append(rows[i])
        print(sname, len(idx), flush=True)
    tests = {}
    for k in K.TESTS:
        t = C['tests'][k]; prep = prep_session(t['raw']); items = []
        for c in t['items']:
            f = feats(prep, c['peak']['hand'], c['peak']['peak_ms'])
            if f is None: continue
            items.append(dict(hand=c['peak']['hand'], t=c['peak']['peak_ms'], g=float(c['peak']['peak_g']), x=np.asarray(c['x'], np.float32), hf=f[0], dense=f[1], pose=f[2], zero=f[3]))
        tests[k] = dict(truth=t['truth'], items=items); print(k, len(items), flush=True)
    joblib.dump(dict(rows=keep, tests=tests), K.OUT/'dataset2.joblib')
    print('rows', len(keep), 'hf dims', keep[0]['hf'].shape, 'dense dims', keep[0]['dense'].shape)


if __name__ == '__main__':
    main()
