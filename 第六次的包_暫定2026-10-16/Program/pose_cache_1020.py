"""Cache heading-free features and raw pose paths for fast experiments.

For every clear training event and every detected peak (2.1 g) of the ten training
recordings and three pretests: heading-free feature vector (heading_free_model_1019)
plus raw absolute yaw / pitch / roll at -120,-80,-40,-20,0,+20,+40 ms. Nothing here
depends on MIDI labels except the training event list itself.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np

PROG = Path(__file__).resolve().parent
sys.path.insert(0, str(PROG))
import heading_free_model_1019 as H

OUT = PROG.parent/'Data/鼓組對準模型_1020'
DTS = (-120, -80, -40, -20, 0, 20, 40)


def pose_path(prep, hand, t):
    rows = prep[hand]['rows']; times = prep[hand]['times']
    out = []
    for dt in DTS:
        i = int(np.argmin(np.abs(times-(t+dt)))); r = rows[i]
        out.append([r['absolute_yaw'], r['absolute_pitch'], r['roll']])
    return np.array(out, dtype=np.float64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = joblib.load(H.first.OLD_CACHE)
    base = joblib.load(H.first.OUT/'特徵快取.joblib')
    det = dict(base['detector'], threshold_g=2.1)
    keep = set(json.loads((H.clear.OUT/'評分前固定_選樣與開頭鼓種核對.json').read_text(encoding='utf-8'))['training_keys'])
    cache = dict(train=[], peaks={}, det=det)
    for s in sessions:
        H.world.enrich(s['hands'], s['raw']); prep = H.world.base.legacy.prepare(s['hands'])
        for r in s['records']:
            if f"{r['session']}:{r['id']}" not in keep: continue
            orig = next((p for p in s['candidates'] if p['hand'] == r['hand'] and p['peak_ms'] == r['center']), None)
            if orig is None:
                avail = min(len(s['hands'][r['hand']])-1, int(np.searchsorted(prep[r['hand']]['times'], r['center']+80)))
                orig = dict(hand=r['hand'], peak_ms=r['center'], available_index=avail)
            cache['train'].append(dict(session=s['name'], hand=r['hand'], drum=r['drum'], t=r['center'],
                                       hf=H.heading_free(prep, orig, False), pose=pose_path(prep, r['hand'], r['center'])))
        ps = H.world.base.detector.detect(s['hands'], det)[0]
        cache['peaks'][s['name']] = [dict(peak=p, hf=x, pose=pose_path(prep, p['hand'], p['peak_ms']))
                                     for p in ps for x in [H.heading_free(prep, p, False)] if x is not None]
        print(s['name'], len(cache['peaks'][s['name']]), flush=True)
    for t in base['tests']:
        k = 'pretest:'+t['session']['name']
        H.world.enrich(t['hands'], t['session']['raw']); prep = H.world.base.legacy.prepare(t['hands'])
        ps = H.world.base.detector.detect(t['hands'], det)[0]
        cache['peaks'][k] = [dict(peak=p, hf=x, pose=pose_path(prep, p['hand'], p['peak_ms']))
                             for p in ps for x in [H.heading_free(prep, p, False)] if x is not None]
        print(k, len(cache['peaks'][k]), flush=True)
    joblib.dump(cache, OUT/'pose_cache.joblib', compress=3)


if __name__ == '__main__':
    main()
