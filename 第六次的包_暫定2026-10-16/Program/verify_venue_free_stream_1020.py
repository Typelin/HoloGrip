"""Verify the 7th-package field Engine (sample-by-sample) against the batch committee.

1. Stream each Raw CSV through seventh_core.Engine exactly like the live console.
2. Compare (hand, peak_ms, drum) with batch predictions from pose_cache (same model).
3. Score streamed pretest output against ghost-cleaned MIDI (MIDI span only).
4. Heading shift: add a constant to every raw yaw -> predictions must be identical.
Writes reference CSVs (batch) used by the package tests.
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path

import joblib
import numpy as np

PROG = Path(__file__).resolve().parent
PACK = Path(r'C:/Users/Typelin_Station/Desktop/HoloGrip/第七次的包_暫定2026-10-20')
sys.path.insert(0, str(PROG)); sys.path.insert(1, str(PACK/'Program'))
OUT = PROG.parent/'Data/鼓組對準模型_1020'
REF = OUT/'場地無關_即時重播驗證'


def stream(core, bundle, config, raw, turn=0.0):
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(bundle, config); out = []; lat = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]
        s = dict(s, absolute_yaw=(s['absolute_yaw']+turn+180) % 360-180)
        frame = dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=(float(row['yaw_deg'])+turn+180) % 360-180, pitch=float(row['pitch_deg']), roll=s['roll'])
        t0 = time.perf_counter(); p, reason = engine.push(h, s, frame)
        if p: lat.append((time.perf_counter()-t0)*1000)
        assert reason == '', reason
        out.extend(p)
    return sorted(out, key=lambda p: (p['peak_ms'], p['hand'])), lat


def main():
    import seventh_core as core
    import pose_experiments_1020 as E
    import export_venue_free_1020 as X
    import gpu_strict_eval_1020 as G
    bundle, config = core.load_components()
    C = joblib.load(OUT/'pose_cache.joblib'); z = G.load()
    REF.mkdir(parents=True, exist_ok=True)
    raws = {k: t['session']['raw'] for k, t in E.TESTS.items()}
    raws.update({s['name']: s['raw'] for s in E.SESS if s['name'] in ('sixth_S08', 'fifth_song03')})
    report = {}
    for name, raw in raws.items():
        # batch reference
        ref = []
        for h in 'LR':
            it = [c for c in C['peaks'][name] if c['peak']['hand'] == h]
            P = np.array([c['pose'] for c in it]); Z = np.array([list(z[name][h][:2])+[0.0]]*len(it))
            Xb = np.hstack([G.sub(np.array([c['hf'] for c in it], dtype=np.float32), 'summary'), G.rot_batch(P, Z)]).astype(np.float32)
            q = core.runtime.probabilities(bundle, Xb, h)
            ref += [dict(hand=h, peak_ms=float(c['peak']['peak_ms']), drum=bundle['classes'][int(v.argmax())], model_score=float(v.max())) for c, v in zip(it, q)]
        ref.sort(key=lambda p: (p['peak_ms'], p['hand']))
        out, lat = stream(core, bundle, config, raw)
        a = [(p['hand'], float(p['peak_ms']), p['drum']) for p in out]; b = [(p['hand'], p['peak_ms'], p['drum']) for p in ref]
        score_diff = max((abs(p['model_score']-r['model_score']) for p, r in zip(out, ref)), default=0.0)
        safe = name.replace(':', '_')
        with open(REF/f'{safe}.csv', 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['hand', 'peak_ms', 'emitted_ms', 'drum', 'model_score']); w.writeheader()
            for p in out: w.writerow({k: p[k] for k in w.fieldnames})
        rec = dict(stream=len(a), batch=len(b), identical=a == b, max_score_diff=score_diff,
                   latency_ms_p50=float(np.percentile(lat, 50)), latency_ms_p99=float(np.percentile(lat, 99)),
                   emitted_minus_peak_ms_max=float(max(p['emitted_ms']-p['peak_ms'] for p in out)))
        if not rec['identical']:
            diff = [(x, y) for x, y in zip(a, b) if x != y][:5]; rec['first_diff'] = [list(map(str, d)) for d in diff]
        if name.startswith('pretest:'):
            t = E.TESTS[name]; g = set(E.GHOST[name]['ghosts'])
            S = dict(t['session'], truth=[x for x in t['session']['truth'] if x['id'] not in g])
            tt = np.array([float(x['t']) for x in S['truth']]); lo, hi = tt.min()-200, tt.max()+200
            sc = E.H.first.scoring.score(S, [p for p in out if lo <= p['peak_ms'] <= hi])[0]
            rec['score'] = dict(correct=sc['correct'], total=sc['total'], rate=round(sc['correct']/sc['total'], 4))
            sigs = [[(p['hand'], p['peak_ms'], p['drum']) for p in out]]
            for turn in (90.0, -137.0):
                o, _ = stream(core, bundle, config, raw, turn); sigs.append([(p['hand'], p['peak_ms'], p['drum']) for p in o])
            rec['heading_shift_identical'] = all(s == sigs[0] for s in sigs)
        report[name] = rec
        print(name, rec, flush=True)
    (REF/'驗證結果.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
