"""Venue-independent drum model: no magnetic/compass heading in any feature.

Same events, clear-sample training list (6,667), external detector, per-hand MLP
256->128 spec and full-Raw scoring as the field model clear_accel256sqrt. Only the
features change: every world-frame vector (linear acceleration, relative rotation,
angular velocity, integrated motion) is re-expressed in the frame of THAT HAND'S OWN
heading at the hit (Rz(-yaw_at_hit)). If every yaw shifts by a constant (new room,
facing another way) the features are bit-for-bit identical, verified below by adding
+90/+180 deg to every raw yaw.
Variants:
  hf      heading-free only
  hf_zero hf + hand heading relative to the recorded drumhead zero (still venue-free:
          zero is taken in the same room/session)
Evaluation: full Raw -> external detector -> features -> classify -> compare MIDI.
Ten training-scope recordings are resubstitution; three pretests are not in training.
"""
from __future__ import annotations
from bisect import bisect_left, bisect_right
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

PROG = Path(__file__).resolve().parent
sys.path.insert(0, str(PROG))
import absolute_world_motion_runtime_1019 as world
import research_pretest_zone_models_1019 as first
import research_clear_single_support_1019 as clear
import pretest_zone_trajectory_runtime_1019 as zone_rt

OUT = PROG.parent/'Data/場地無關模型_1019'
CONTRACT = 'hologrip_heading_free_1019_v1'


def heading_free(prepared, peak, with_zero):
    rows = prepared[peak['hand']]['rows']; times = prepared[peak['hand']]['times']
    available = min(int(peak['available_index']), len(rows)-1)
    idx = peak.get('peak_index')
    if idx is None or not 0 <= int(idx) <= available or abs(rows[int(idx)]['t']-peak['peak_ms']) > .01:
        delta = np.abs(times[:available+1]-peak['peak_ms']); ties = np.flatnonzero(delta == delta.min())
        idx = max(ties, key=lambda i: rows[int(i)]['mag'])
    idx = int(idx); before = 240
    sc = rows[idx]['sensor_t']
    lo = bisect_left(rows, sc-before-30, 0, available+1, key=lambda r: r['sensor_t'])
    hi = bisect_right(rows, sc+80, 0, available+1, key=lambda r: r['sensor_t'])
    sel = rows[lo:hi]
    if len(sel) < 6: return None
    clock = np.array([r['sensor_t']-sc for r in sel]); _, u = np.unique(clock, return_index=True)
    sel = [sel[int(i)] for i in u]; clock = clock[u]
    if len(clock) < 6 or np.any(np.diff(clock) <= 0): return None
    grid = np.arange(-before, 81., 10.); clipped = np.clip(grid, clock[0], clock[-1])
    acc = np.array([[r['ax'], r['ay'], r['az']] for r in sel])
    ang = np.array([[r['absolute_yaw'], r['absolute_pitch'], r['roll']] for r in sel])
    body = np.column_stack([np.interp(grid, clock, acc[:, j]) for j in range(3)])
    rot = Slerp(clock, Rotation.from_euler('ZYX', ang, degrees=True))(clipped)
    c = int(np.argmin(np.abs(grid)))
    yaw_c = rot[c].as_euler('ZYX', degrees=True)[0]
    H = Rotation.from_euler('Z', -yaw_c, degrees=True)          # remove compass heading
    lin = H.apply(rot.apply(body)-np.array([0., 0., 1.]))
    rel = H.apply((rot*rot[c].inv()).as_rotvec())
    av = H.apply(np.vstack([np.zeros(3), (rot[1:]*rot[:-1].inv()).as_rotvec()/.01]))
    d = np.radians(rot.as_euler('ZYX', degrees=True)[:, 0]-yaw_c)
    yaw_path = np.unwrap((d+np.pi) % (2*np.pi)-np.pi)          # heading change within stroke, wrap-safe
    yaw_path -= yaw_path[c]
    tilt = rot.as_euler('ZYX', degrees=True)[:, 1:]                                 # pitch/roll: gravity-referenced
    mag = np.linalg.norm(body, axis=1); scale = max(1., float(np.linalg.norm(acc, axis=1).max()))
    snap = np.array([int(np.argmin(abs(grid-t))) for t in [-160, -120, -80, -40, 0, 40, 80]])
    wave = np.column_stack([lin/scale, rel/np.pi, av/50., (mag-1)/scale, yaw_path/np.pi, tilt/90.])
    integ = np.cumsum(lin, axis=0)*.01/scale
    scal = [np.log1p(scale), float(peak['hand'] == 'R')]
    for f0, f1 in [(-before, -120), (-120, -40), (-40, 40), (40, 80)]:
        m = (grid >= f0) & (grid <= f1); a, w, g = lin[m], av[m], mag[m]
        scal.extend(np.r_[a.mean(0)/scale, a.std(0)/scale, w.mean(0)/50., w.std(0)/50., np.percentile(g, [10, 50, 90])/scale, np.abs(g-1).sum()/scale])
    v = np.r_[wave.ravel(), integ[snap].ravel(), scal]
    if with_zero:
        zrel = np.radians(rows[idx]['yaw'])        # recorded calibrated yaw = raw yaw - drumhead zero yaw
        v = np.r_[v, np.sin(zrel), np.cos(zrel), rows[idx]['pitch']/90.]
    assert np.isfinite(v).all()
    return v.astype(np.float32)


def train(X, y, hands, w):
    models = {}
    for h in 'LR':
        m = hands == h
        p = Pipeline([('scaler', StandardScaler()), ('model', MLPClassifier((256, 128), alpha=.3, random_state=20261019,
                      max_iter=450, n_iter_no_change=25, tol=1e-4, batch_size=128))])
        p.fit(X[m], y[m], model__sample_weight=w[m]); models[h] = p
    return models


def classify(bundle, peaks, vectors):
    return zone_rt.classify(bundle, peaks, vectors)


def features_for_peaks(prep, peaks, with_zero):
    sel, vec = [], []
    for p in peaks:
        x = heading_free(prep, p, with_zero)
        if x is not None: sel.append(p); vec.append(x)
    return sel, vec


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter()
    sessions = joblib.load(first.OLD_CACHE)
    base = joblib.load(first.OUT/'特徵快取.joblib')
    keep = set(json.loads((clear.OUT/'評分前固定_選樣與開頭鼓種核對.json').read_text(encoding='utf-8'))['training_keys'])
    det = base['detector']
    preps = {}
    for s in sessions:
        world.enrich(s['hands'], s['raw']); preps[s['name']] = world.base.legacy.prepare(s['hands'])
    tests = base['tests']
    for t in tests:
        world.enrich(t['hands'], t['session']['raw']); preps['test:'+t['session']['name']] = world.base.legacy.prepare(t['hands'])
    report = {}
    for variant, wz in [('hf', False), ('hf_zero', True)]:
        X, y, hands, keys = [], [], [], []
        for s in sessions:
            prep = preps[s['name']]
            for r in s['records']:
                k = f"{r['session']}:{r['id']}"
                if k not in keep: continue
                orig = next((p for p in s['candidates'] if p['hand'] == r['hand'] and p['peak_ms'] == r['center']), None)
                if orig is None:
                    avail = min(len(s['hands'][r['hand']])-1, int(np.searchsorted(prep[r['hand']]['times'], r['center']+80)))
                    orig = dict(hand=r['hand'], peak_ms=r['center'], available_index=avail)
                x = heading_free(prep, orig, wz); assert x is not None
                X.append(x); y.append(r['drum']); hands.append(r['hand']); keys.append(k)
        X = np.array(X); y = np.array(y); hands = np.array(hands)
        cell = Counter(zip(hands, y)); w = np.array([1/np.sqrt(cell[(h, z)]) for h, z in zip(hands, y)]); w /= w.mean()
        with threadpool_limits(limits=1):
            models = train(X, y, hands, w)
        bundle = dict(runtime_contract=CONTRACT, mode='per_hand', models=models, classes=sorted(set(y)), variant=variant,
                      with_zero=wz, training_events=len(y), detector_config=det, compass_heading_used=False)
        joblib.dump(bundle, OUT/f'{variant}.joblib', compress=3)
        # 10 training-scope recordings, full pipeline
        scores = {}
        for s in sessions:
            peaks, _ = world.base.detector.detect(s['hands'], det)
            sel, vec = features_for_peaks(preps[s['name']], peaks, wz)
            pred = classify(bundle, sel, vec)
            sc, _, _ = first.training.assess(s, pred)
            scores[s['name']] = dict(correct=sc['correct'], total=sc['total'], rate=round(sc['rate'], 4),
                                     missed=sc.get('missed'), wrong=sc.get('wrong_drum'), extra=sc.get('unmatched_extra'))
        # three pretests, full pipeline (not in training)
        pre = {}
        for t in tests:
            sel, vec = features_for_peaks(preps['test:'+t['session']['name']], t['peaks'], wz)
            pred = classify(bundle, sel, vec)
            sc = first.scoring.score(t['session'], pred)[0]
            pre[t['session']['name']] = dict(correct=sc['correct'], total=sc['total'], rate=round(sc['rate'], 4))
        # heading invariance: shift every raw yaw by +90 and +180 deg on the pretests
        inv = {}
        for shift in (90.0, 180.0):
            same = tot = 0
            for t in tests:
                prep0 = preps['test:'+t['session']['name']]
                prep1 = {h: dict(rows=[dict(r, absolute_yaw=((r['absolute_yaw']+shift+180) % 360)-180) for r in v['rows']], times=v['times']) for h, v in prep0.items()}
                a = classify(bundle, *features_for_peaks(prep0, t['peaks'], wz))
                b = classify(bundle, *features_for_peaks(prep1, t['peaks'], wz))
                same += sum(x['drum'] == z['drum'] for x, z in zip(a, b)); tot += len(a)
            inv[f'+{int(shift)}deg'] = f'{same}/{tot} identical'
        report[variant] = dict(training_events=len(y), resubstitution=scores, pretests=pre, heading_shift_check=inv)
        print(json.dumps({variant: report[variant]}, ensure_ascii=False), flush=True)
    report['seconds'] = round(time.perf_counter()-t0)
    (OUT/'結果.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
