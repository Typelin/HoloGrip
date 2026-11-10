"""Score every labelled training take of every day (songs, four-drum, single-hand) through a package's real field engine.

Answer key = the take's training rows (the labelled hits the model was trained on: hand, drum, peak time);
notes left unlabelled / excluded (ghost notes, glove dropout after cut_ms, ambiguous) are not scored.
The model was trained on these takes, so this is a development (in-sample) score, not a blind test.
Replay is causal and sample by sample, like score.py / score_takes.py (same Engine and anchor setting).
(score_takes.py does the same for one new day using its label files; this covers R4..R10 from the training rows.)

  HOLOGRIP_ROUNDS=out/rounds_r10pre.json python score_alltakes.py --code 第十一次的包_暫定 --tag final11_takes
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--code', default='11-10的包')
ap.add_argument('--tag', default='alltakes')
ap.add_argument('--workers', type=int, default=6)
ARGS = ap.parse_args() if __name__ == '__main__' else ap.parse_args([])

import csv
import joblib
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/ARGS.code/'Program'), str(HERE)]
import seventh_core as core
assert Path(core.__file__).resolve().parent == (ROOT/ARGS.code/'Program').resolve()
import rounds_data as RD
import verify_package_1027 as VP

L5 = RD.L5
P7 = ROOT/'第七次的包_暫定2026-10-20/Data/實驗_姿態模型自動標記_20261020/實驗快取.joblib'
_ENG = {}


def stream(raw):
    if 'b' not in _ENG: _ENG['b'], _ENG['c'] = core.load_components()
    bundle, config = _ENG['b'], _ENG['c']
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(bundle, config, snare_anchor=core.runtime.PRETEST_SNARE_ANCHOR); out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]
        p, reason = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        out.extend(p)
    return sorted(out, key=lambda p: (p['peak_ms'], p['hand']))


def work(job):
    name, raw, truth = job
    P = stream(raw); ok, n, cells = VP.score(truth, P)
    return name, dict(ok=ok, n=n, rate=round(ok/n, 4), cells={k: [v[0], v[1], v[2]] for k, v in cells.items()})


def raw_paths():
    C = joblib.load(P7); raws = {k: v['raw'] for k, v in C['sessions'].items()}
    raws.update({n: d['raw_csv'] for n, d in L5.r8_docs().items()})
    for R, c in RD.config().items(): raws.update({n: d['raw_csv'] for n, d in RD.docs_of(R, c).items()})
    return raws


def main():
    rows, *_ = RD.setup()
    truth = defaultdict(list)
    for r in rows: truth[r['session']].append(dict(t=float(r['t']), drum=r['drum'], hand=r['hand']))
    raws = raw_paths()
    missing = sorted(set(truth)-set(raws)); assert not missing, f'no raw for {missing}'
    jobs = [(n, str(raws[n]), sorted(truth[n], key=lambda g: g['t'])) for n in sorted(truth)]
    print('takes', len(jobs), 'hits', sum(len(j[2]) for j in jobs), flush=True)
    res = {}
    with Pool(ARGS.workers) as pool:
        for name, r in pool.imap_unordered(work, jobs):
            r['day'] = L5.day_of(name); res[name] = r
            print(r['day'], name, r['ok'], '/', r['n'], r['rate'], flush=True)
    (RD.OUT/f'score_{ARGS.tag}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
