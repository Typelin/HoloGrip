"""Median zero-relative yaw/pitch at the hit per pretest drum (answer key from the pretest MIDI), vs the model's
drum positions. Shows whether a hand's whole kit is shifted, or two drums overlap.

  python pose_blocks.py <day> <pretest name>     e.g. R10 pretest:R10  (HOLOGRIP_ROUNDS may point to the config)
"""
import bisect, csv, sys
DAY, NAME = sys.argv[1], sys.argv[2]   # read before imports that rewrite sys.argv
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'第十次的包_暫定'/'Program'), str(HERE)]
import seventh_core as core
import rounds_data as RD
import truth_clean_1027 as TC


def main():
    RD.setup(); L5, H = RD.L5, RD.H
    raw = str(RD.pretest_names(DAY, RD.config()[DAY])[NAME])
    truth, _ = TC.dedupe(L5.TRUTH[NAME], H.CAND[NAME])
    src = list(csv.DictReader(open(raw, encoding='utf-8-sig', newline='')))
    bundle, config = core.load_components(); rt = core.Engine(bundle, config, snare_anchor=5).runtime
    for h in 'LR':
        rows = [r for r in src if r['hand'] == h]; ts = [float(r['song_time_ms']) for r in rows]; per = {}
        for g in truth:
            if g['hand'] != h: continue
            i = min(bisect.bisect_left(ts, g['t']), len(rows)-1)
            seg = rows[max(i-8, 0):i+9]   # hit = acceleration peak within +-80 ms of the MIDI note
            r = max(seg, key=lambda r: float(r['accel_magnitude_g']))
            per.setdefault(g['drum'], []).append((float(r['cal_yaw_deg']), float(r['cal_pitch_deg'])))
        print(h, 'drum      today yaw/pitch (median)   model yaw/pitch')
        for d in rt.classes:
            if d not in per: continue
            m = np.median(per[d], 0); tp = rt.tm_pose[h][rt.classes.index(d)]
            print(f'  {d:9s} {m[0]:8.1f} {m[1]:7.1f}        {tp[0]:8.1f} {tp[1]:7.1f}')


if __name__ == '__main__':
    main()
