"""Evaluate the anchor safeguard (anchor_guard.py) on the package field engine, causal sample-by-sample replay.

Sets:
  pre_full   : every pretest, full model (out/model_r10c.joblib)
  pre_lodo   : pretests of R4..R9, each by out/model_lodor10c_<day>.joblib (day unseen)
  takes_full : every labelled take R4..R10 (in-sample, development score)
  takes_lodo : labelled takes of R4..R9, each by the model that never saw that day
Per take and hand it records the anchor decision, the truth drums of the first 5 emitted hits and any revocation.

  HOLOGRIP_ROUNDS=out/rounds_r10pre.json HG_GUARD=off python guard_eval.py --tag base
  HOLOGRIP_ROUNDS=out/rounds_r10pre.json HG_GUARD='{"after":15,"margin":0.05}' python guard_eval.py --tag g15
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--code', default='11-10的包')
ap.add_argument('--sets', default='pre_full,pre_lodo,takes_full,takes_lodo')
ap.add_argument('--tag', default='base')
ap.add_argument('--workers', type=int, default=6)
ap.add_argument('--anchor', type=int, default=5)
ARGS = ap.parse_args() if __name__ == '__main__' else ap.parse_args([])

import joblib
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/ARGS.code/'Program'), str(HERE)]
import seventh_core as core
import anchor_guard
anchor_guard.install_stats_from_env(core.runtime); anchor_guard.install_from_env(core.runtime)
import rounds_data as RD
import verify_package_1027 as VP
import truth_clean_1027 as TC
FB, L5, H = RD.FB, RD.L5, RD.H
FULL = RD.OUT/'model_r10c.joblib'
LODO_DAYS = ['R4', 'R5', 'R6', 'R7', 'R8', 'R9']
_B = {}


def bundle(path):
    if path not in _B:
        if 'config' not in _B: _B['config'] = core.load_components()[1]
        b = joblib.load(path); core.runtime.attach_boosters(b); _B[path] = b
    return _B[path], _B['config']


def stream(model, raw):
    b, config = bundle(model)
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(b, config, snare_anchor=ARGS.anchor); out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]
        p, reason = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        out.extend(p)
    return sorted(out, key=lambda p: (p['peak_ms'], p['hand'])), engine.runtime


def work(job):
    kind, name, model, raw, truth = job
    P, rt = stream(model, raw); ok, n, cells = VP.score(truth, P)
    inv = {j: i for i, j in VP.V.match(truth, P).items()}
    first = {}
    for h in 'LR':
        idx = [j for j, p in enumerate(P) if p['hand'] == h][:ARGS.anchor]
        first[h] = [truth[inv[j]]['drum'] if j in inv else None for j in idx]
    anc = {h: dict(anchored=rt.anchored[h], drum=rt.classes[rt.anchored_drum[h]] if rt.anchored_drum[h] is not None else None,
                   revoked=getattr(rt, 'revoked', {}).get(h, False), revoked_at=getattr(rt, 'revoked_at', {}).get(h),
                   revoked_drum=getattr(rt, 'revoked_drum', {}).get(h), first_truth=first[h]) for h in 'LR'}
    return kind, name, dict(ok=ok, n=n, rate=round(ok/n, 4), anchor=anc, cells={k: [v[0], v[1]] for k, v in cells.items()})


def jobs():
    rows, *_ = RD.setup(); out = []
    sets = ARGS.sets.split(',')
    if {'pre_full', 'pre_lodo'} & set(sets):
        raws = {k: VP.V.C['tests'][k]['raw'] for k in VP.V.TESTS}; raws[L5.PRE] = str(L5.PRE_RAW)   # = score.raw_paths()
        for R, c in RD.config().items(): raws.update({n: str(p) for n, p in RD.pretest_names(R, c).items()})
        for D in FB.DAYS:
            for nm in FB.PRETESTS[D]:
                truth, _ = TC.dedupe(L5.TRUTH[nm], H.CAND[nm])
                if 'pre_full' in sets: out.append(('pre_full', nm, str(FULL), str(raws[nm]), truth))
                if 'pre_lodo' in sets and D in LODO_DAYS: out.append(('pre_lodo', nm, str(RD.OUT/f'model_lodor10c_{D}.joblib'), str(raws[nm]), truth))
    if {'takes_full', 'takes_lodo'} & set(sets):
        import score_alltakes as SA
        truth = defaultdict(list)
        for r in rows: truth[r['session']].append(dict(t=float(r['t']), drum=r['drum'], hand=r['hand']))
        raws = SA.raw_paths()
        for nm in sorted(truth):
            tr = sorted(truth[nm], key=lambda g: g['t']); D = L5.day_of(nm)
            if 'takes_full' in sets: out.append(('takes_full', nm, str(FULL), str(raws[nm]), tr))
            if 'takes_lodo' in sets and D in LODO_DAYS: out.append(('takes_lodo', nm, str(RD.OUT/f'model_lodor10c_{D}.joblib'), str(raws[nm]), tr))
    return out


def main():
    J = jobs(); print('jobs', len(J), 'guard', os.environ.get('HG_GUARD', 'off'), flush=True)
    res = defaultdict(dict)
    with Pool(ARGS.workers) as pool:
        for kind, name, r in pool.imap_unordered(work, J, chunksize=1):
            res[kind][name] = r
            rv = {h: a['revoked_drum'] for h, a in r['anchor'].items() if a['revoked']}
            print(kind, name, r['ok'], '/', r['n'], r['rate'], {h: a['drum'] for h, a in r['anchor'].items()}, 'REVOKED' if rv else '', rv or '', flush=True)
    summ = {}
    for kind, d in res.items():
        clean = {k: v for k, v in d.items() if k != 'pretest:R10b'}
        ok = sum(v['ok'] for v in clean.values()); n = sum(v['n'] for v in clean.values())
        cells = [c for v in clean.values() for c in v['cells'].values()]
        summ[kind] = dict(ok=ok, n=n, rate=round(ok/max(n, 1), 4), takes=len(clean), cells_failed=sum(c[0] < 0.85*c[1] for c in cells) if kind.startswith('pre') else None,
                          revoked=sum(a['revoked'] for v in clean.values() for a in v['anchor'].values()))
    out = dict(tag=ARGS.tag, guard=os.environ.get('HG_GUARD', 'off'), summary=summ, results=res)
    (RD.OUT/f'guard_{ARGS.tag}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
