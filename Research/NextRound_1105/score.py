"""Score pretests through the real field engine (sample by sample, causal), per take and per hand x drum cell.

Answer key: pretest MIDI with electronic-drum double triggers removed (truth_clean_1027.dedupe).
A cell passes at >= 85% (17/20). Every pretest take is scored separately (never merged).

  python score.py                                    # 9th package code + model, every pretest (R4..R8 + enabled new days)
  python score.py --model out/model_tenth.joblib     # another bundle on the same field code
  python score.py --lodo                             # each day scored by out/model_lodo_<day>.joblib
  python score.py --live 第九次的包_暫定 --days R9    # the predictions the field program actually made that day
Writes out/score_<tag>.json and out/score_<tag>.md.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--code', default='第九次的包_暫定', help='package whose Program/ runs the replay')
ap.add_argument('--model', default='', help='bundle to replay (default: the package model, fingerprint-checked)')
ap.add_argument('--lodo', action='store_true', help='score day D with out/model_lodo_<D>.joblib')
ap.add_argument('--lodo-prefix', default='lodo', help='leave-one-day-out models are out/model_<prefix>_<day>.joblib')
ap.add_argument('--live', default='', help='package whose Data/模型實測/*/predictions.csv is scored (no replay)')
ap.add_argument('--days', default='', help='comma list of days (default all)')
ap.add_argument('--tag', default='')
ap.add_argument('--refs', default='', help='also write each replay to out/<refs>/<take>.csv (package test references)')
ARGS = ap.parse_args() if __name__ == '__main__' else ap.parse_args([])   # spawned worker processes re-import this module

import numpy as np
import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/ARGS.code/'Program'), str(HERE)]
import seventh_core as core
assert Path(core.__file__).resolve().parent == (ROOT/ARGS.code/'Program').resolve()
import rounds_data as RD
FB, L5, K, H = RD.FB, RD.L5, RD.K, RD.H
import verify_package_1027 as VP
import truth_clean_1027 as TC

PASS = 0.85


def stream(bundle, config, raw):
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(bundle, config, snare_anchor=core.runtime.PRETEST_SNARE_ANCHOR); out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]
        p, reason = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        assert reason == '', reason; out.extend(p)
    return sorted(out, key=lambda p: (p['peak_ms'], p['hand']))


def live_predictions(raw):
    with open(Path(raw).parent/'predictions.csv', encoding='utf-8-sig', newline='') as f:
        return [dict(hand=r['hand'], peak_ms=float(r['peak_ms']), drum=r['drum']) for r in csv.DictReader(f)]


def raw_paths():
    raws = {k: VP.V.C['tests'][k]['raw'] for k in VP.V.TESTS}; raws[L5.PRE] = str(L5.PRE_RAW)
    for R, c in RD.config().items(): raws.update({n: str(p) for n, p in RD.pretest_names(R, c).items()})
    return raws


def load_bundle(path, config):
    b = joblib.load(path); core.runtime.attach_boosters(b)
    assert b['detector_config'] == config, 'detector parameters differ from the package'
    return b


def table(res):
    lines = ['| 前測 | 正確 | 正確率 | 未達 17/20 的格子 |', '|---|---:|---:|---|']
    for nm, r in res['takes'].items():
        bad = '、'.join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(r['cells'].items()) if v[0] < PASS*v[1]) or '無'
        lines.append(f"| {nm} | {r['ok']}/{r['n']} | {100*r['rate']:.1f}% | {bad} |")
    lines.append(f"| 合計 | {res['ok']}/{res['n']} | {100*res['rate']:.1f}% | 未達 {res['cells_failed']}/{res['cells']} 格 |")
    return '\n'.join(lines)


def main():
    RD.setup()
    days = ARGS.days.split(',') if ARGS.days else list(FB.DAYS)
    raws = raw_paths(); config = None; bundle = None
    if not ARGS.live:
        bundle, config = core.load_components()
        if ARGS.model: bundle = load_bundle(Path(ARGS.model) if Path(ARGS.model).is_absolute() else HERE/ARGS.model, config)
    mode = 'live' if ARGS.live else ('lodo' if ARGS.lodo else ('model' if ARGS.model else 'package'))
    res = dict(mode=mode, code=ARGS.code, model=ARGS.model, live=ARGS.live, takes={}, ok=0, n=0)
    for D in days:
        b = load_bundle(RD.OUT/f'model_{ARGS.lodo_prefix}_{D}.joblib', config) if ARGS.lodo else bundle
        for nm in FB.PRETESTS[D]:
            P = live_predictions(raws[nm]) if ARGS.live else stream(b, config, raws[nm])
            if ARGS.refs:
                ref = RD.OUT/ARGS.refs; ref.mkdir(exist_ok=True)
                with open(ref/f"{nm.replace(':', '_')}.csv", 'w', encoding='utf-8-sig', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['hand', 'peak_ms', 'emitted_ms', 'peak_g', 'drum', 'model_score']); w.writeheader()
                    for p in P: w.writerow({q: p[q] for q in w.fieldnames})
            truth, dropped = TC.dedupe(L5.TRUTH[nm], H.CAND[nm])
            ok, n, cells = VP.score(truth, P)
            res['takes'][nm] = dict(day=D, ok=ok, n=n, rate=round(ok/n, 4), double_triggers_removed=dropped,
                                    cells={k: [v[0], v[1], v[2]] for k, v in cells.items()})
            res['ok'] += ok; res['n'] += n
            print(D, nm, ok, '/', n, round(ok/n, 4), flush=True)
    res['rate'] = round(res['ok']/max(res['n'], 1), 4)
    allc = [v for r in res['takes'].values() for v in r['cells'].values()]
    res['cells'] = len(allc); res['cells_failed'] = sum(v[0] < PASS*v[1] for v in allc)
    tag = ARGS.tag or mode+('_'+Path(ARGS.model).stem if ARGS.model else '')
    (RD.OUT/f'score_{tag}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    md = table(res); (RD.OUT/f'score_{tag}.md').write_text(md+'\n', encoding='utf-8')
    print(md, flush=True)


if __name__ == '__main__':
    main()
