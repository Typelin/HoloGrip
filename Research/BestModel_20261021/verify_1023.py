"""Replay Raw CSVs through the EXACT 8th-package field Engine (sample by sample, like 1_測前次) and score:
6 pretests (14 cells, hand from glove peak) + per-recording MIDI score for the pretests and a few songs.
Also checks heading invariance (all yaw + 90 deg) on one pretest. Writes reference CSVs for the tests."""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, joblib
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT/'第八次的包_暫定2026-10-23'
sys.path.insert(0, str(PACK/'Program'))
import seventh_core as core
OUT = Path(__file__).resolve().parent/'out'
C = joblib.load(ROOT/'第七次的包_暫定2026-10-20/Data/實驗_姿態模型自動標記_20261020/實驗快取.joblib')
TRUTH = joblib.load(OUT/'pretest_truth_hands.joblib')
TESTS = ['pretest:round4', 'pretest:round5', 'pretest:round6', 'seventh_01', 'seventh_12', 'seventh_15']
DRUMS = ['小鼓', '高音 Tom', '中音 Tom', '落地 Tom', 'Hi-Hat', 'Crash', 'Ride']


def stream(bundle, config, raw, turn=0.0):
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(bundle, config); out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]; s = dict(s, absolute_yaw=(s['absolute_yaw']+turn+180) % 360-180)
        frame = dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=(float(row['yaw_deg'])+turn+180) % 360-180, pitch=float(row['pitch_deg']), roll=s['roll'])
        p, reason = engine.push(h, s, frame); assert reason == '', reason; out.extend(p)
    return sorted(out, key=lambda p: (p['peak_ms'], p['hand'])), engine


def match(truth, P):
    tt = np.array([g['t'] for g in truth]); pt = np.array([p['peak_ms'] for p in P]); dt = np.abs(tt[:, None]-pt[None, :])
    hm = np.array([[g.get('hand') is not None and g['hand'] != p['hand'] for p in P] for g in truth])
    cost = np.hstack([np.where(dt <= 80, dt+300*hm, 1e9), np.full((len(tt), len(tt)), 1e6)])
    r, c = linear_sum_assignment(cost); return {i: j for i, j in zip(r, c) if j < len(pt) and cost[i, j] <= 380}


def main():
    bundle, config = core.load_components(); cells = defaultdict(Counter); rec = {}; ref = OUT/'replay_1023'; ref.mkdir(exist_ok=True)
    for k in TESTS:
        P, eng = stream(bundle, config, C['tests'][k]['raw']); truth = TRUTH[k]; pairs = match(truth, P)
        tt = [g['t'] for g in truth]; lo, hi = min(tt)-200, max(tt)+200
        ok = 0
        for i, g in enumerate(truth):
            j = pairs.get(i); o = 'miss' if j is None else ('hand' if g['hand'] and P[j]['hand'] != g['hand'] else P[j]['drum'])
            if g['hand']: cells[(g['hand'], g['drum'])][o] += 1
            ok += o == g['drum']
        extra = sum(1 for j, p in enumerate(P) if j not in set(pairs.values()) and lo <= p['peak_ms'] <= hi)
        rec[k] = dict(correct=ok, total=len(truth), rate=round(ok/len(truth), 4), outputs_in_span=sum(1 for p in P if lo <= p['peak_ms'] <= hi), extra=extra, adapted=dict(eng.runtime.adapted))
        with open(ref/f"{k.replace(':', '_')}.csv", 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['hand', 'peak_ms', 'emitted_ms', 'peak_g', 'drum', 'model_score']); w.writeheader()
            for p in P: w.writerow({q: p[q] for q in w.fieldnames})
        print(k, rec[k], flush=True)
    acc = {f'{h}|{d}': cells[(h, d)][d]/max(sum(cells[(h, d)].values()), 1) for h in 'LR' for d in DRUMS}
    print(' '.join(f"{k}{v*100:.0f}" for k, v in acc.items()), '| mean', round(np.mean(list(acc.values()))*100, 1), 'min', round(min(acc.values())*100, 1))
    P0, _ = stream(bundle, config, C['tests']['seventh_15']['raw']); P1, _ = stream(bundle, config, C['tests']['seventh_15']['raw'], 90.0)
    inv = [(p['hand'], p['peak_ms'], p['drum']) for p in P0] == [(p['hand'], p['peak_ms'], p['drum']) for p in P1]
    print('heading +90 identical:', inv)
    (OUT/'verify_1023.json').write_text(json.dumps(dict(pretests=rec, cells=acc, heading_invariant=inv), ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
