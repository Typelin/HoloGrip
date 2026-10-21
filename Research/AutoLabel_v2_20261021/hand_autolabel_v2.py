"""General hand auto-labeler v2 (offline annotation aid, experiment only).

For every MIDI event in a song recording, predict which glove (L/R) played it from
glove evidence around the aligned event time:
  * per-hand activity (|accel|-1 g) maxima in +-30 / +-60 ms, timing of the maximum,
    pre-motion activity, nearest independent peak and whether a closer MIDI event owns it;
  * per-hand zero-relative rotation of the nearest detected hit (no compass heading);
  * MIDI context: drum, velocity relative to the drum's median in the recording,
    simultaneous / nearby events and the partner drum.
XGBoost (GPU) is validated leave-one-recording-out against the complete historical
label backups (round 4/5/6 songs). Human-confirmed labels are reported separately
because they are the hard cases that needed a person. A confidence threshold is
chosen on held-out recordings only. Weak-hit (ghost suspect) rule is checked against
human exclusions. Nothing formal is modified; outputs go to this folder.
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT/'第六次的包_暫定2026-10-16'; P7 = ROOT/'第七次的包_暫定2026-10-20'
EXP = P7/'Data/實驗_姿態模型自動標記_20261020'
OUT = Path(__file__).resolve().parent/'out'; OUT.mkdir(exist_ok=True)
DRUMS = ['小鼓', '高音 Tom', '中音 Tom', '落地 Tom', 'Hi-Hat', 'Crash', 'Ride']
HIST = ['fourth_song03', 'fifth_song01', 'fifth_song02', 'fifth_song03', 'sixth_S02', 'sixth_S08']
TODAY = ['seventh_03', 'seventh_04', 'seventh_05', 'seventh_13', 'seventh_14']


def is_human(l):
    src = l.get('label_source', l.get('source', ''))
    return bool(l.get('human_confirmed') is True or l.get('human_sequence_review') is True or src.startswith('human_'))


def hand_series(s):
    out = {}
    for h, a in s['raw_signals']['hands'].items():
        a = np.asarray(a, dtype=float); out[h] = (a[:, 0], a[:, 1])
    return out


def window_max(ts, act, t0, w):
    i, j = np.searchsorted(ts, t0-w), np.searchsorted(ts, t0+w, 'right')
    if j <= i: return 0.0, np.nan
    k = i+int(np.argmax(act[i:j])); return float(act[k]), float(ts[k]-t0)


def window_mean(ts, act, a, b):
    i, j = np.searchsorted(ts, a), np.searchsorted(ts, b, 'right')
    return float(act[i:j].mean()) if j > i else 0.0


def features(s):
    ev = [e for e in s['events'] if not e.get('unmapped') and e.get('name') in DRUMS]
    ser = hand_series(s)
    peaks = defaultdict(list)
    for p in s['raw_signals']['peaks']: peaks[p['hand']].append(p)
    ptime = {h: np.array([p['t'] for p in v]) for h, v in peaks.items()}
    items = defaultdict(list)
    for c in s['items']: items[c['peak']['hand']].append(c)
    itime = {h: np.array([c['peak']['peak_ms'] for c in v]) for h, v in items.items()}
    et = np.array([e['csv_t'] for e in ev])
    vel_med = {d: np.median([e['vel'] for e in ev if e['name'] == d] or [1]) for d in DRUMS}
    rows = []
    for k, e in enumerate(ev):
        t = e['csv_t']; f = {}
        near = [q for q in ev[max(0, k-6):k+7] if q is not e and abs(q['csv_t']-t) <= 80]
        f['n_near40'] = sum(abs(q['csv_t']-t) <= 40 for q in near); f['n_near80'] = len(near)
        partner = min(near, key=lambda q: abs(q['csv_t']-t)) if near else None
        f['partner_dt'] = abs(partner['csv_t']-t) if partner else 999.
        for d in DRUMS:
            f['drum_'+d] = float(e['name'] == d); f['partner_'+d] = float(partner is not None and partner['name'] == d)
        f['vel'] = e['vel']; f['vel_rel'] = e['vel']/max(vel_med[e['name']], 1)
        for h in 'LR':
            ts, act = ser[h]
            for w in (30, 60, 100):
                m, dt = window_max(ts, act, t, w); f[f'{h}_max{w}'] = m; f[f'{h}_dt{w}'] = dt
            f[f'{h}_pre'] = window_mean(ts, act, t-200, t-80); f[f'{h}_post'] = window_mean(ts, act, t+80, t+200)
            pt = ptime.get(h, np.array([]))
            if len(pt):
                j = int(np.argmin(np.abs(pt-t))); p = peaks[h][j]; dt = p['t']-t
                f[f'{h}_pk_dt'] = dt if abs(dt) <= 100 else np.nan; f[f'{h}_pk_val'] = p['value'] if abs(dt) <= 100 else 0.
                f[f'{h}_pk_prom'] = p['prominence'] if abs(dt) <= 100 else 0.
                owner = np.min(np.abs(et-p['t'])); f[f'{h}_pk_owned_by_other'] = float(abs(dt) <= 100 and owner < abs(dt)-1)
            else:
                f[f'{h}_pk_dt'] = np.nan; f[f'{h}_pk_val'] = 0.; f[f'{h}_pk_prom'] = 0.; f[f'{h}_pk_owned_by_other'] = 0.
            it = itime.get(h, np.array([]))
            rot = np.full(21, np.nan)
            if len(it):
                j = int(np.argmin(np.abs(it-t)))
                if abs(it[j]-t) <= 60 and 'x' in items[h][j]: rot = np.asarray(items[h][j]['x'])[-21:]
            for q in range(21): f[f'{h}_rot{q}'] = rot[q]
        f['diff_max60'] = f['L_max60']-f['R_max60']
        f['logratio60'] = np.log((f['L_max60']+.05)/(f['R_max60']+.05))
        f['logratio30'] = np.log((f['L_max30']+.05)/(f['R_max30']+.05))
        rows.append(f)
    names = list(rows[0]); X = np.array([[r[n] for n in names] for r in rows], dtype=np.float32)
    return ev, X, names


def labels_for(name, ev):
    old = {s['name']: s for s in joblib.load(P6/'Data/鼓面歸零_跨次完整CSV考試_20261019_v1/特徵快取.joblib')}
    b = json.loads(Path(old[name]['backup']).read_text(encoding='utf-8-sig')); L = b['labels']
    y, hum, state = [], [], []
    for e in ev:
        l = L.get(str(e['id']), {}); m = l.get('mark')
        if l.get('needs_recheck'): m = 'recheck'
        y.append(1 if m == 'L' else 0 if m == 'R' else -1); hum.append(is_human(l)); state.append(m or 'none')
    return np.array(y), np.array(hum), np.array(state)


def fit(X, y, seed=0):
    m = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=.05, subsample=.85, colsample_bytree=.7,
                          reg_lambda=2., device='cuda', tree_method='hist', random_state=seed, eval_metric='logloss')
    m.fit(X, y); return m


def main():
    C = joblib.load(EXP/'實驗快取.joblib')
    data = {}
    for n in HIST:
        ev, X, names = features(C['sessions'][n]); y, hum, st = labels_for(n, ev)
        data[n] = dict(ev=ev, X=X, y=y, hum=hum, st=st)
    # leave-one-recording-out
    loro = {}
    for n in HIST:
        tr = [m for m in HIST if m != n]
        Xtr = np.vstack([data[m]['X'][data[m]['y'] >= 0] for m in tr]); ytr = np.concatenate([data[m]['y'][data[m]['y'] >= 0] for m in tr])
        p = np.mean([fit(Xtr, ytr, sd).predict_proba(data[n]['X'])[:, 1] for sd in range(3)], 0)
        loro[n] = p
    # threshold sweep on held-out predictions
    sweep = []
    for th in [.5, .8, .9, .95, .97, .98, .99, .995]:
        tot = dict(valid=0, ok=0, hum_valid=0, hum_ok=0, cover=0, events=0, excl_hit=0, per={})
        for n in HIST:
            d = data[n]; p = loro[n]; conf = np.maximum(p, 1-p); pred = (p >= .5).astype(int); sel = conf >= th
            v = sel & (d['y'] >= 0); hv = v & d['hum']
            ok = int((pred[v] == d['y'][v]).sum()); hok = int((pred[hv] == d['y'][hv]).sum())
            tot['valid'] += int(v.sum()); tot['ok'] += ok; tot['hum_valid'] += int(hv.sum()); tot['hum_ok'] += hok
            tot['cover'] += int(sel.sum()); tot['events'] += len(p); tot['excl_hit'] += int((sel & (d['st'] == 'exclude')).sum())
            tot['per'][n] = dict(acc=round(ok/max(v.sum(), 1), 4), n=int(v.sum()), hum_acc=round(hok/max(hv.sum(), 1), 4), hum_n=int(hv.sum()))
        tot['acc'] = round(tot['ok']/tot['valid'], 4); tot['hum_acc'] = round(tot['hum_ok']/max(tot['hum_valid'], 1), 4)
        tot['coverage'] = round(tot['cover']/tot['events'], 4); tot['threshold'] = th
        hum_total = sum(int(((data[n]['y'] >= 0) & data[n]['hum']).sum()) for n in HIST)
        tot['hum_coverage'] = round(tot['hum_valid']/hum_total, 4)
        sweep.append(tot); print(json.dumps({k: v for k, v in tot.items() if k != 'per'}, ensure_ascii=False), flush=True)
    joblib.dump(dict(loro=loro, data=data, names=names), OUT/'loro.joblib')
    (OUT/'threshold_sweep.json').write_text(json.dumps(sweep, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
