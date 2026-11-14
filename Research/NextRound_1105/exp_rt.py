"""Runtime-parameter experiments on the 11-17 field engine (no retraining; model weights untouched).

Every take is replayed sample by sample through 11-17的包/Program (causal, peak + 80 ms) with these knobs patched:
  decay     forgetting factor of the per-drum offset statistics per confident hit (1.0 = field: running mean of all
            earlier hits; < 1 follows slow direction drift / body sway). Also used by the zero-guard replay.
  alpha, conf, minh             bundle['adaptation'] (offset strength, confidence to enter stats, hits before use)
  pw, ps                        position prior weight / sigma (deg)
  c_deg, c_score, c_ms          pose continuity limits
  anchor                        automatic snare anchor size
  floor                         lower bound of the position-prior kernel (0 = field)
Sets:
  A  17 pretests (R4..R13), model g6 (pretests are never in training)            -> selection
  B  17 pretests, each day by the leave-one-day-out model lodog2_<day>             -> generalisation check
  C  11/12 drills + song 1 by lodog2_R13 (11/12 never in that model): 4 fast sway drills (reference),
     4 controlled drills, song 1 (hand checked on single notes only)               -> drift stress test
  python exp_rt.py <grid name> [--sets A,B,C] [--workers 14]
Writes out/exp_rt_<grid>.json and prints one line per config."""
from __future__ import annotations
import argparse, csv, itertools, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; PKG = ROOT/'11-17的包'
os.environ.setdefault('HOLOGRIP_ROUNDS', 'out/rounds_R13.json')
sys.path[:0] = [str(PKG/'Program'), str(HERE), str(ROOT/'Research/AutoLabel_1027')]
DEFAULT = dict(floor=0.0, decay=1.0, alpha=1.0, conf=0.8, minh=20, pw=0.5, ps=12.0, c_deg=16.0, c_score=0.9, c_ms=600.0, anchor=5)
DRILLS = {'L_M': ('115527', '11：12 Tom 2.mid', 'sway'), 'R_M': ('120107', '11：12Tom 2 (右手）.mid', 'sway'),
          'L_M_R': ('120523', '11：12Tom Ride (左手).mid', 'sway'), 'R_M_R': ('120924', '11：12 Tom Ride (右手).mid', 'sway'),
          'L_H_M_R': ('121424', '11：12Tom1＼2Ride (左手).mid', 'ctrl'), 'R_H_M_R': ('121855', '11：12 Tom1：2 Ride (右手).mid', 'ctrl'),
          'L_M_R_G': ('122405', '11：12Tom2 Ride Ftom(左手).mid', 'ctrl'), 'R_M_R_G': ('122833', '11：12Tom Ride Ftom(右手).mid', 'ctrl'),
          '歌曲1': ('113853', '11：12歌曲1(修正.mid', 'song')}

GRIDS = {
    'base': [{}],
    'decay': [dict(decay=d) for d in (1.0, 0.995, 0.99, 0.98, 0.97, 0.95, 0.93, 0.9)],
    'adapt': [dict(alpha=a, conf=c, minh=m) for a in (0.8, 1.0, 1.2) for c in (0.7, 0.8, 0.9) for m in (10, 20)],
    'prior': [dict(pw=w, ps=s) for w in (0.0, 0.25, 0.5, 0.75, 1.0) for s in (8.0, 12.0, 16.0, 20.0) if not (w == 0 and s != 12.0)],
    'cont': [dict(c_deg=d, c_score=s, c_ms=m) for d in (0.0, 10.0, 16.0, 24.0) for s in (0.8, 0.9, 0.95) for m in (400.0, 600.0)
             if not (d == 0 and (s, m) != (0.9, 600.0))],
    'anchor': [dict(anchor=a) for a in (0, 3, 5, 8)],
}


def jobs_for(sets):
    import rounds_data as RD, truth_clean_1027 as TC, autolabel_1027 as AL
    FB, L5, H, VP = RD.FB, RD.L5, RD.H, None
    RD.setup()
    import verify_package_1027 as VP
    raws = {k: VP.V.C['tests'][k]['raw'] for k in VP.V.TESTS}; raws[L5.PRE] = str(L5.PRE_RAW)
    for R, c in RD.config().items(): raws.update({n: str(p) for n, p in RD.pretest_names(R, c).items()})
    out = []
    for D in FB.DAYS:
        for nm in FB.PRETESTS[D]:
            truth, _ = TC.dedupe(L5.TRUTH[nm], H.CAND[nm])
            if 'A' in sets: out.append(('A', D, nm, raws[nm], str(HERE/'out/model_g6.joblib'), truth))
            if 'B' in sets: out.append(('B', D, nm, raws[nm], str(RD.OUT/f'model_lodog2_{D}.joblib'), truth))
    if 'C' in sets:
        M = ROOT/'11-12的包/今天/雲端_11-12'
        for nm, (key, mid, kind) in DRILLS.items():
            raw = next((ROOT/'11-12的包/Data/Raw').glob(f'*_{key}.csv'))
            cand = L5.hit_cand.work((raw.stem, str(raw), 1.5, {}))[1]; ct = np.array(sorted(x['t'] for x in cand))
            ev = [e for e in AL.legacy.midi(M/mid)['events'] if e['vel'] >= AL.GHOST_VEL and e['name'] in RD.K.DRUMS]
            mt = np.array([e['t'] for e in ev])
            s, off, _ = max(((s,)+tuple(AL.align(mt*s, ct)) for s in np.r_[1.0, np.arange(0.95, 1.06, 0.0005)]), key=lambda x: x[2])
            truth = []
            for e in ev:
                t = e['t']*s+off; near = [x for x in cand if abs(x['t']-t) <= 60]
                truth.append(dict(id=e['id'], t=t, drum=e['name'], hand=max(near, key=lambda x: x['g'])['hand'] if near else None))
            if kind == 'song':
                ts = np.array([g['t'] for g in truth])
                for g in truth:
                    if int(np.sum(np.abs(ts-g['t']) <= 40)) > 1: g['hand'] = None
            out.append(('C', kind, nm, str(raw), str(RD.OUT/'model_lodog2_R13.joblib'), truth))
    return out


_CACHE = {}


def _bundle(path):
    import joblib, seventh_core as core
    if path not in _CACHE:
        if len(_CACHE) >= 2: _CACHE.pop(next(iter(_CACHE)))
        if 'cfg' not in _CACHE: _CACHE['cfg'] = core.load_components()[1]
        b = joblib.load(path); core.runtime.attach_boosters(b); _CACHE[path] = b
    return _CACHE[path], _CACHE['cfg']


def work(job):
    cfg, (S, D, nm, raw, model, truth) = job
    import seventh_core as core, verify_package_1027 as VP
    rt = core.runtime; p = {**DEFAULT, **cfg}
    if not hasattr(rt, '_Stats0'):
        rt._Stats0 = rt._Stats; rt.CONT0 = dict(rt.CONTINUITY)
        class DStats(rt._Stats0):
            LAM = 1.0
            def force(self, rot, d):
                if DStats.LAM < 1.0: self.sum *= DStats.LAM; self.n *= DStats.LAM
                self.sum[d] += rot; self.n[d] += 1; self.total += 1
        rt._Stats = DStats
    rt._Stats.LAM = p['decay']
    if not hasattr(rt, 'PP0'):
        rt.PP0 = rt.StreamEngine._position_prior
        def pp(self, hand, pr, yp):
            if not self.position: return pr
            d = yp[None, :]-self.tm_pose[hand]; d[:, 0] = (d[:, 0]+180) % 360-180
            k = np.maximum(np.exp(-(d**2).sum(1)/(2*self.position['sigma_deg']**2)), rt.FLOOR)
            q = pr*k**self.position['weight']+1e-12
            return q/q.sum()
        rt.StreamEngine._position_prior = pp
    rt.FLOOR = p['floor']
    rt.CONTINUITY.clear(); rt.CONTINUITY.update(rt.CONT0, max_deg=p['c_deg'], below_score=p['c_score'], max_ms=p['c_ms'])
    b0, config = _bundle(model)
    b = dict(b0); b['adaptation'] = dict(b0['adaptation'], alpha=p['alpha'], confidence=p['conf'], min_confident_hits=p['minh'])
    if b0.get('position_prior'): b['position_prior'] = dict(b0['position_prior'], weight=p['pw'], sigma_deg=p['ps'])
    if p['pw'] == 0: b['position_prior'] = None
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(b, config, snare_anchor=p['anchor']); P = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]
        out, _ = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        P.extend(out)
    P = [dict(hand=q['hand'], t=q['peak_ms'], peak_ms=q['peak_ms'], drum=q['drum']) for q in P]
    ok, n, cells = VP.score(truth, P)
    return json.dumps(cfg, sort_keys=True), S, D, nm, ok, n, {k: v[:2] for k, v in cells.items()}


def summarise(rows):
    r = {}
    for key, S, D, nm, ok, n, cells in rows:
        x = r.setdefault(key, {}); s = x.setdefault(S, dict(ok=0, n=0, failed=0, takes={}))
        s['ok'] += ok; s['n'] += n; s['takes'][nm] = [ok, n]; s.setdefault('cells', {})[nm] = cells
        if S != 'C': s['failed'] += sum(v[0] < 0.85*v[1] for v in cells.values())
        else:
            k = s.setdefault('kind', {}).setdefault(D, [0, 0]); k[0] += ok; k[1] += n
    return r


def line(key, x):
    out = [f'{key:60s}']
    for S in 'ABC':
        if S not in x: continue
        s = x[S]
        if S == 'C': out.append('C ' + ' '.join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(s['kind'].items())))
        else: out.append(f"{S} {s['ok']}/{s['n']} fail{s['failed']}")
    return ' | '.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('grid'); ap.add_argument('--sets', default='A,B,C'); ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--cfgs', default='', help='JSON list of configs instead of a named grid')
    ap.add_argument('--takes', default='', help='comma list of take names to keep')
    a = ap.parse_args(); sets = a.sets.split(',')
    cfgs = json.loads(a.cfgs) if a.cfgs else GRIDS[a.grid]
    base = jobs_for(sets); t0 = time.time()
    if a.takes: base = [j for j in base if j[2] in a.takes.split(',')]
    # group by model so each worker's bundle cache hits
    jobs = sorted([(c, j) for c in cfgs for j in base], key=lambda cj: (cj[1][4], json.dumps(cj[0], sort_keys=True)))
    with Pool(a.workers) as pool: rows = list(pool.imap_unordered(work, jobs, chunksize=2))
    res = summarise(rows)
    (HERE/f'out/exp_rt_{a.grid}.json').write_text(json.dumps(dict(sets=sets, res=res), ensure_ascii=False, indent=1), encoding='utf-8')
    for key in sorted(res, key=lambda k: -res[k].get('A', res[k].get('C', {'ok': 0}))['ok']): print(line(key, res[key]))
    print(f'{len(jobs)} jobs, {time.time()-t0:.0f} s', flush=True)
