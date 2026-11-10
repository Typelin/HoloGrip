"""Sweep the inference parameters stored in the model bundle (no retraining) through the real field engine.

Each setting is scored on every pretest twice: with the full-training bundle and with the leave-one-day-out
bundles (out/model_<prefix>_<day>.joblib). Parameters (bundle fields, read by venue_free_runtime_1023):
  prior_w / prior_sigma : position_prior weight / sigma_deg
  gate                  : hit_gate threshold
  raw_share             : committee weight of the raw members (re-centred members get the rest)
  conf / minh           : per-drum offset correction confidence / min_confident_hits

  python score_params.py --model out/model_ex.joblib --lodo-prefix lodoex --grid default --tag params_ex
Writes out/score_<tag>.json (per setting: full + LODO totals, failed cells, per pretest).
"""
from __future__ import annotations
import argparse, copy, csv, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--code', default='第十次的包_暫定')
ap.add_argument('--model', default='out/model_ex.joblib')
ap.add_argument('--lodo-prefix', default='lodoex')
ap.add_argument('--grid', default='default')
ap.add_argument('--tag', default='params_ex')
ARGS = ap.parse_args() if __name__ == '__main__' else ap.parse_args([])

import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/ARGS.code/'Program'), str(HERE)]
import seventh_core as core
assert Path(core.__file__).resolve().parent == (ROOT/ARGS.code/'Program').resolve()
import rounds_data as RD
FB, L5, H = RD.FB, RD.L5, RD.H
import verify_package_1027 as VP
import truth_clean_1027 as TC

PASS = 0.85
BASE = dict(prior_w=0.5, prior_sigma=12.0, gate=0.4, raw_share=0.4, conf=0.8, minh=20)
GRIDS = {'default': [dict()]
         + [dict(prior_w=w) for w in (0.0, 0.25, 0.75, 1.0)]
         + [dict(prior_sigma=s) for s in (8.0, 10.0, 15.0, 20.0)]
         + [dict(gate=g) for g in (0.3, 0.35, 0.45, 0.5)]
         + [dict(raw_share=r) for r in (0.0, 0.25, 0.5, 0.6)]
         + [dict(conf=c) for c in (0.7, 0.9)]
         + [dict(minh=m) for m in (10, 30)]}


def apply(b, s):
    b = dict(b)
    b['position_prior'] = dict(b['position_prior'], weight=s['prior_w'], sigma_deg=s['prior_sigma'])
    b['hit_gate'] = dict(b['hit_gate'], threshold=s['gate'])
    b['adaptation'] = dict(b['adaptation'], confidence=s['conf'], min_confident_hits=s['minh'])
    keys = list(b['portable'])
    half = len(keys)//2   # members are ordered raw first, then re-centred (build_model: train(tr)+train(rc))
    b['member_weights'] = {k: (s['raw_share']/half if i < half else (1-s['raw_share'])/(len(keys)-half)) for i, k in enumerate(keys)}
    return b


def stream(bundle, config, raw):
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(bundle, config, snare_anchor=core.runtime.PRETEST_SNARE_ANCHOR); out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]
        p, reason = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        assert reason == '', reason; out.extend(p)
    return sorted(out, key=lambda p: (p['peak_ms'], p['hand']))


def raw_paths():
    raws = {k: VP.V.C['tests'][k]['raw'] for k in VP.V.TESTS}; raws[L5.PRE] = str(L5.PRE_RAW)
    for R, c in RD.config().items(): raws.update({n: str(p) for n, p in RD.pretest_names(R, c).items()})
    return raws


def load(path):
    b = joblib.load(path); core.runtime.attach_boosters(b); return b


def run(bundles, config, s, raws):
    takes = {}; ok_t = n_t = failed = 0
    for D in FB.DAYS:
        b = apply(bundles[D], s); core.runtime.attach_boosters(b)
        for nm in FB.PRETESTS[D]:
            P = stream(b, config, raws[nm])
            truth, _ = TC.dedupe(L5.TRUTH[nm], H.CAND[nm])
            ok, n, cells = VP.score(truth, P)
            f = sum(v[0] < PASS*v[1] for v in cells.values())
            takes[nm] = dict(ok=ok, n=n, failed=f, cells={k: [v[0], v[1]] for k, v in cells.items()})
            ok_t += ok; n_t += n; failed += f
    return dict(ok=ok_t, n=n_t, rate=round(ok_t/n_t, 4), failed=failed, takes=takes)


def main():
    RD.setup(); _, config = core.load_components(); raws = raw_paths()
    full = load(HERE/ARGS.model); lodo = {D: load(RD.OUT/f'model_{ARGS.lodo_prefix}_{D}.joblib') for D in FB.DAYS}
    out_path = RD.OUT/f'score_{ARGS.tag}.json'; res = []
    for delta in GRIDS[ARGS.grid]:
        s = dict(BASE, **delta)
        r = dict(setting=delta or 'base', full=run({D: full for D in FB.DAYS}, config, s, raws), lodo=run(lodo, config, s, raws))
        res.append(r)
        print(json.dumps(dict(setting=r['setting'], full=f"{r['full']['ok']}/{r['full']['n']} fail {r['full']['failed']}",
                              lodo=f"{r['lodo']['ok']}/{r['lodo']['n']} fail {r['lodo']['failed']}"), ensure_ascii=False), flush=True)
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
