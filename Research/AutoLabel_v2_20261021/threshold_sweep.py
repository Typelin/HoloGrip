"""Detector threshold sweep: recall, extra hits and drum accuracy by peak-g band.

Detect at a low threshold (1.6 g) once, build the deployed 104 features for every peak, classify with
the deployed field committee, then evaluate each threshold T by keeping peaks with peak_g >= T.
(Raising the threshold only removes peaks in this detector; refractory interactions are re-checked by
re-running the detector at each T for the hit counts.)
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, joblib, warnings
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
P7 = ROOT/'第七次的包_暫定2026-10-20'
sys.path[:0] = [str(P7/'Program')]
import seventh_core as core
import venue_free_runtime_1020 as rt
import causal_hit_detector_1016 as detector
EXP = P7/'Data/實驗_姿態模型自動標記_20261020'
C = joblib.load(EXP/'實驗快取.joblib')
bundle, cfg = core.load_components()
TH = [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5]


def run(raw, T):
    hands = rt.world.read_raw(raw); prep = rt.world.base.legacy.prepare(hands)
    peaks, _ = detector.detect(hands, dict(cfg, threshold_g=T)); out = []
    for p in peaks:
        s = prep[p['hand']]['rows'][p['available_index']]
        x = rt.feature(prep, p, (s['absolute_yaw']-s['yaw']+180) % 360-180, s['absolute_pitch']-s['pitch'])
        if x is None: continue
        q = rt.probabilities(bundle, x[None, :], p['hand'])[0]
        out.append(dict(t=p['peak_ms'], g=p['peak_g'], drum=bundle['classes'][int(q.argmax())], score=float(q.max())))
    return out


def match(truth, preds):
    tt = np.array([g['t'] for g in truth]); pt = np.array([p['t'] for p in preds])
    dt = np.abs(tt[:, None]-pt[None, :]); cost = np.hstack([np.where(dt <= 80, dt, 1e9), np.full((len(tt), len(tt)), 1e6)])
    r, c = linear_sum_assignment(cost); return {i: j for i, j in zip(r, c) if j < len(pt) and cost[i, j] <= 80}


def main():
    sets = {k: (C['tests'][k]['truth'], C['tests'][k]['raw']) for k in ['seventh_01', 'seventh_12', 'seventh_15', 'pretest:round4', 'pretest:round5', 'pretest:round6']}
    for k in ['seventh_13', 'seventh_14']:
        s = C['sessions'][k]; sets[k] = ([dict(t=e['csv_t'], drum=e['name']) for e in s['events'] if e.get('name') and not e.get('unmapped') and not e.get('out_of_scope')], s['raw'])
    res = defaultdict(dict); band = defaultdict(lambda: dict(n=0, matched=0, correct=0))
    for name, (truth, raw) in sets.items():
        tt = np.array([g['t'] for g in truth]); lo, hi = tt.min()-200, tt.max()+200
        for T in TH:
            preds = [p for p in run(raw, T) if lo <= p['t'] <= hi]
            pairs = match(truth, preds); inv = {j: i for i, j in pairs.items()}
            ok = sum(1 for i, j in pairs.items() if preds[j]['drum'] == truth[i]['drum'])
            res[name][T] = dict(total=len(truth), matched=len(pairs), correct=ok, extra=len(preds)-len(pairs))
            if T == min(TH):
                for j, p in enumerate(preds):
                    b = 'lt1.9' if p['g'] < 1.9 else '1.9-2.1' if p['g'] < 2.1 else '2.1-2.3' if p['g'] < 2.3 else '2.3-2.6' if p['g'] < 2.6 else 'ge2.6'
                    band[b]['n'] += 1
                    if j in inv: band[b]['matched'] += 1; band[b]['correct'] += preds[j]['drum'] == truth[inv[j]]['drum']
        print(name, {T: (v['correct'], v['matched'], v['extra']) for T, v in res[name].items()}, flush=True)
    agg = {T: {q: sum(res[n][T][q] for n in res) for q in ('total', 'matched', 'correct', 'extra')} for T in TH}
    for T, v in agg.items(): print('ALL', T, v, 'rate', round(v['correct']/v['total'], 4), 'precision', round(v['correct']/(v['matched']+v['extra']), 4))
    for b, v in band.items(): print('BAND', b, v, 'drum acc of matched', round(v['correct']/max(v['matched'], 1), 3), 'false-hit share', round(1-v['matched']/max(v['n'], 1), 3))
    (Path(__file__).parent/'out/threshold_sweep_detector.json').write_text(json.dumps(dict(per=res, all=agg, band=band), ensure_ascii=False, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
