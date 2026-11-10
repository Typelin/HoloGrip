"""Screen every labelled training take (all days) through the real field engine, each with models that never
saw it:  (a) the leave-one-day-out model of its own day  out/model_<prefix>_<day>.joblib   (--mode lodo)
         (b) one fixed model for every take                                              (--mode model --model ...)
Answer key = the take's training rows. Per take: score, per-hand score, kind (song / four_drum / single_hand).
A very low score = the take disagrees with all other days (candidate for exclusion).

  HOLOGRIP_ROUNDS=out/rounds_1112.json python screen_all_1112.py --prefix lodoc12nolhm --tag lodo_nolhm
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('--code', default='11-10的包'); ap.add_argument('--mode', default='lodo'); ap.add_argument('--prefix', default='lodoc12nolhm')
ap.add_argument('--model', default=''); ap.add_argument('--tag', default='lodo_nolhm'); ap.add_argument('--only', default='', help='comma list of take-name substrings')
ARGS = ap.parse_args() if __name__ == '__main__' else ap.parse_args([a for a in sys.argv[1:]] if '--tag' in sys.argv else [])
import joblib
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/ARGS.code/'Program'), str(HERE)]
import seventh_core as core
import rounds_data as RD
import verify_package_1027 as VP
L5 = RD.L5
_B = {}


def bundle(path):
    if path not in _B:
        if 'config' not in _B: _B['config'] = core.load_components()[1]
        b = joblib.load(path); core.runtime.attach_boosters(b); _B.clear(); _B['config'] = core.load_components()[1]; _B[path] = b
    return _B[path], _B['config']


def work(job):
    name, model, raw, truth = job
    b, config = bundle(model)
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(b, config, snare_anchor=core.runtime.PRETEST_SNARE_ANCHOR); P = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]
        p, _ = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        P.extend(p)
    P.sort(key=lambda p: (p['peak_ms'], p['hand'])); ok, n, cells = VP.score(truth, P)
    hand = {h: [sum(v[0] for k, v in cells.items() if k.startswith(h)), sum(v[1] for k, v in cells.items() if k.startswith(h))] for h in 'LR'}
    rt = engine.runtime
    return name, dict(ok=ok, n=n, rate=round(ok/n, 4), hand=hand, zero_rot=dict(rt.zero_rot), anchored={h: rt.anchored[h] for h in 'LR'},
                      cells={k: [v[0], v[1], v[2]] for k, v in cells.items()})


def main():
    import score_alltakes as SA
    rows, *_ = RD.setup(); truth = defaultdict(list); kind = {}
    for r in rows: truth[r['session']].append(dict(t=float(r['t']), drum=r['drum'], hand=r['hand'])); kind[r['session']] = r.get('kind', '')
    raws = SA.raw_paths(); only = [s for s in ARGS.only.split(',') if s]; J = []
    for nm in sorted(truth):
        if only and not any(s in nm for s in only): continue
        D = L5.day_of(nm); m = RD.OUT/f'model_{ARGS.prefix}_{D}.joblib' if ARGS.mode == 'lodo' else HERE/ARGS.model
        if not m.exists(): print('no model for', nm, m.name, flush=True); continue
        J.append((nm, str(m), str(raws[nm]), sorted(truth[nm], key=lambda g: g['t'])))
    J.sort(key=lambda j: j[1]); res = {}
    with Pool(6) as pool:
        for nm, r in pool.imap(work, J, chunksize=1):
            r['day'] = L5.day_of(nm); r['kind'] = kind.get(nm, ''); res[nm] = r
            print(r['day'], nm, r['ok'], '/', r['n'], r['rate'], 'L', r['hand']['L'], 'R', r['hand']['R'], 'zero', r['zero_rot'], flush=True)
    (RD.OUT/f'screen_{ARGS.tag}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('total', sum(r['ok'] for r in res.values()), '/', sum(r['n'] for r in res.values()))


if __name__ == '__main__':
    main()
