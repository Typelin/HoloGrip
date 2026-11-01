"""Training rows, hit-gate labels and pretest answer keys for rounds 4-8 plus any new field day.

Rounds 4-8 come from Research/AutoLabel_1027 (final_1027.setup, unchanged). Each new day listed in
rounds.json (enabled) adds:
  * labelled recordings  Data/自動標記_<date>/*_自動標記_<date>.json  (autolabel_day.py)  -> training rows
  * pretests             Data/模型實測/S<date>*/*_raw_100hz_*.csv + the pretest MIDI    -> exam only
Names: '<R>:<take>' for labelled takes, 'pretest:<R>' (or 'pretest:<R>#<k>' for several pretests).
Pretests never enter training or the hit gate (they are appended to K.TESTS).
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AL27 = ROOT/'Research/AutoLabel_1027'
sys.path[:0] = [str(AL27), str(HERE)]
import final_1027 as FB
L5, K, H = FB.L5, FB.K, FB.H
OUT = HERE/'out'; OUT.mkdir(exist_ok=True)


def config():
    cfg = json.loads((HERE/os.environ.get('HOLOGRIP_ROUNDS', 'rounds.json')).read_text(encoding='utf-8'))
    return {R: c for R, c in cfg.items() if not R.startswith('_') and c.get('enabled')}


def label_dir(c):
    if c.get('label_dir'): return Path(c['label_dir']) if Path(c['label_dir']).is_absolute() else HERE/c['label_dir']
    return ROOT/c['package']/'Data'/f"自動標記_{c['date']}"


def pretest_raws(c):
    if c.get('pretest_raw'): return [Path(p) if Path(p).is_absolute() else ROOT/c['package']/p for p in c['pretest_raw']]
    return sorted((ROOT/c['package']/'Data/模型實測').glob(f"S{c['date']}*/S{c['date']}_*_raw_100hz_*.csv"))


def pretest_names(R, c):
    raws = pretest_raws(c)
    return {(f'pretest:{R}' if len(raws) == 1 else f'pretest:{R}#{k+1}'): r for k, r in enumerate(raws)}


_EXTRA = {}
_base_day_of = L5.day_of


def day_of(name):
    for R in _EXTRA:
        if name.startswith(f'{R}:') or name == f'pretest:{R}' or name.startswith(f'pretest:{R}#'): return R
    return _base_day_of(name)


L5.day_of = day_of; L5.LF.day_of = day_of


def docs_of(R, c):
    tag = f"_自動標記_{c['date']}"
    return {f"{R}:{p.name.split(tag)[0]}": json.loads(p.read_text(encoding='utf-8')) for p in sorted(label_dir(c).glob(f'*{tag}.json'))}


def rows_of(R, docs):
    rows = L5.r8_rows(docs)
    for r in rows: r['group'] = R
    return rows


def truth_of(midi_path, cand):
    """Pretest answer key: MIDI notes (velocity >= 10) aligned to the glove candidates; hand = strongest candidate within 60 ms."""
    import autolabel_1027 as AL
    ev = [e for e in AL.legacy.midi(midi_path)['events'] if e['vel'] >= AL.GHOST_VEL and e['name'] in K.DRUMS]
    ct = np.array(sorted(x['t'] for x in cand))
    off, rate = AL.align(np.array([e['t'] for e in ev]), ct)
    truth = []
    for e in ev:
        t = e['t']+off; near = [x for x in cand if abs(x['t']-t) <= 60]
        truth.append(dict(id=e['id'], t=t, drum=e['name'], hand=max(near, key=lambda x: x['g'])['hand'] if near else None))
    return truth, dict(offset_ms=round(off, 1), align_rate=round(rate, 4), hands=dict(Counter(g['hand'] for g in truth)))


def load_round(R, c):
    """Candidates (1.5 g), training rows and pretest truth of one new day, cached in out/round_<R>.joblib."""
    docs = docs_of(R, c); pre = pretest_names(R, c)
    key = dict(docs=sorted(docs), pretests={k: str(v) for k, v in pre.items()},
               mtime=max([p.stat().st_mtime for p in label_dir(c).glob('*.json')]+[Path(v).stat().st_mtime for v in pre.values()], default=0))
    cf = OUT/f'round_{R}.joblib'
    if cf.exists():
        D = joblib.load(cf)
        if D['key'] == key: return D
    jobs = [(n, str(p), 1.5, {}) for n, p in pre.items()]+[(n, d['raw_csv'], 1.5, {}) for n, d in docs.items()]
    with Pool(6) as pool: cand = dict(pool.imap_unordered(L5.hit_cand.work, jobs))
    T = {}
    for name, d in docs.items():
        used = [dict(t=float(e['csv_ms']), drum=d['labels'][str(e['id'])]['drum'], hand=d['labels'][str(e['id'])]['mark'])
                for e in d['events'] if d['labels'][str(e['id'])]['mark'] in ('L', 'R')]
        amb = [float(e['csv_ms']) for e in d['events'] if d['labels'][str(e['id'])]['mark'] not in ('L', 'R')]
        T[name] = (used, amb)
    mids = c['pretest_midi'] if isinstance(c['pretest_midi'], list) else [c['pretest_midi']]
    assert len(mids) == len(pre), f'{R}: {len(pre)} pretest recordings but {len(mids)} pretest MIDI files'
    truth, info = {}, {}
    for (n, _), m in zip(sorted(pre.items()), mids):
        truth[n], info[n] = truth_of(Path(c['midi_dir'])/m, cand[n])
    D = dict(key=key, cand=cand, rows=rows_of(R, docs), T=T, truth=truth, truth_info=info)
    joblib.dump(D, cf)
    return D


def setup(extra=None):
    """Everything final_1027.setup returns, plus the enabled new days. Returns (rows, names, Lb, T, info)."""
    rows, names, Lb, T, _ = FB.setup()
    cfg = config()
    if extra is not None: cfg = {R: c for R, c in cfg.items() if R in extra}
    info = {}
    for R, c in cfg.items():
        _EXTRA[R] = c
        D = load_round(R, c)
        H.CAND.update(D['cand']); T.update(D['T']); rows = rows+D['rows']
        for n, tr in D['truth'].items():
            L5.TRUTH[n] = tr
            if n not in K.TESTS: K.TESTS.append(n)
        FB.PRETESTS[R] = sorted(D['truth'])
        if R not in FB.DAYS: FB.DAYS.append(R)
        info[R] = dict(rows=len(D['rows']), takes=len(D['T']), pretests=D['truth_info'],
                       by_hand_drum={f'{h}|{d}': n for (h, d), n in sorted(Counter((r['hand'], r['drum']) for r in D['rows']).items())})
    names = list(H.CAND)
    for n in names:
        if n not in Lb and n not in K.TESTS: Lb[n] = L5.gate_labels(n, T)
    return rows, names, Lb, T, info
