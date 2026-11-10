"""Score the labelled (non-pretest) takes of a new day through the real field engine, per take and per hand x drum.

Answer key: the day's labels (autolabel_day.py), only notes marked L/R; excluded / review notes are not scored.
Same engine settings as the pretest (automatic anchor on the first 5 hits per hand).

  python score_takes.py --code 第九次的包_暫定 --day R9 --tag takes9_R9    # package that never saw the day
  python score_takes.py --code 第十次的包_暫定 --day R9 --tag takes10_R9   # package trained on the day
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--code', default='第九次的包_暫定')
ap.add_argument('--day', default='R9')
ap.add_argument('--tag', default='')
ARGS = ap.parse_args() if __name__ == '__main__' else ap.parse_args([])

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/ARGS.code/'Program'), str(HERE)]
import seventh_core as core
assert Path(core.__file__).resolve().parent == (ROOT/ARGS.code/'Program').resolve()
import rounds_data as RD
import verify_package_1027 as VP
import csv


def stream(bundle, config, raw):
    """Same as score.stream (not imported: score.py would pin the 9th package code)."""
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(bundle, config, snare_anchor=core.runtime.PRETEST_SNARE_ANCHOR); out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]
        p, reason = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        assert reason == '', reason; out.extend(p)
    return sorted(out, key=lambda p: (p['peak_ms'], p['hand']))


def main():
    RD.setup(); bundle, config = core.load_components()
    c = RD.config()[ARGS.day]; docs = RD.docs_of(ARGS.day, c)
    res = dict(code=ARGS.code, day=ARGS.day, takes={})
    for nm, d in docs.items():
        used = [dict(t=float(e['csv_ms']), drum=d['labels'][str(e['id'])]['drum'], hand=d['labels'][str(e['id'])]['mark'])
                for e in d['events'] if d['labels'][str(e['id'])]['mark'] in ('L', 'R')]
        P = stream(bundle, config, d['raw_csv'])
        ok, n, cells = VP.score(used, P)
        res['takes'][nm] = dict(kind=d['kind'], ok=ok, n=n, rate=round(ok/n, 4), cells={k: [v[0], v[1], v[2]] for k, v in cells.items()})
        print(nm, d['kind'], ok, '/', n, round(ok/n, 4), flush=True)
    tag = ARGS.tag or f"takes_{ARGS.code.split('的包')[0]}_{ARGS.day}"
    (RD.OUT/f'score_{tag}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
