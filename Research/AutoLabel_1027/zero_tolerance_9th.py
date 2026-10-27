"""How much zeroing error does the 9th package tolerate?

Replays real pretests through the 9th package field engine with the drumhead zero deliberately rotated
by delta degrees in yaw (as if the hand pointed elsewhere while zeroing): only the zero changes, the
motion during play is untouched. Scores with the cleaned answer key.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_9th as V9
core, FB, VP, TC, L5 = V9.core, V9.FB, V9.VP, V9.TC, V9.L5


def stream_zero_shift(bundle, config, raw, delta):
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(bundle, config, snare_anchor=core.runtime.PRETEST_SNARE_ANCHOR); out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]
        s = dict(s, yaw=(s['yaw']-delta+180) % 360-180)   # absolute heading unchanged; zero = absolute_yaw - yaw moves by +delta
        frame = dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll'])
        p, reason = engine.push(h, s, frame); assert reason == '', reason; out.extend(p)
    return sorted(out, key=lambda p: (p['peak_ms'], p['hand']))


def main():
    FB.setup(); bundle, config = core.load_components()
    raws = {k: V9.V.C['tests'][k]['raw'] for k in V9.V.TESTS}; raws[L5.PRE] = str(L5.PRE_RAW)
    res = {}
    for nm in [L5.PRE, 'pretest:round5', 'seventh_01']:
        T = TC.dedupe(L5.TRUTH[nm], FB.H.CAND[nm])[0]
        for delta in [0, 20, -20, 45, -45, 90]:
            P = stream_zero_shift(bundle, config, raws[nm], delta)
            ok, n, cells = VP.score(T, P)
            res[f'{nm}|{delta}'] = dict(ok=ok, n=n, rate=round(ok/n, 4))
            print(nm, f'歸零偏 {delta:+d}°', f'{ok}/{n}', round(ok/n*100, 1), flush=True)
            (HERE/'out/zero_tolerance_9th.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
