"""Side-effect fixes for the 8th package, simulated exactly like venue_free_runtime_1023 but with options.
Items (1.7 g detections + features) for all recordings are cached; the 8th-package bundle is used.
Options:
  rescue: min score, allowed cymbals, min peak g, isolation (no own-hand peak >= 2.1 g within W ms)
  offset: min distinct confident drums (each >= k hits) before correcting, max offset norm (clamp)
Reports: pretest 14 cells; songs (correct, extra); single-hand takes (correct)."""
import json, sys, itertools
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, joblib
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'第八次的包_暫定2026-10-27/Program'))
import seventh_core as core
import venue_free_runtime_1023 as R
from verify_1023 import TRUTH, match
import full_report as F
import common as K
import soft_hit_experiment as S
from h5_cymbal_rescue import items_at

SONGS = ['seventh_03', 'seventh_04', 'seventh_05', 'seventh_13', 'seventh_14', 'fifth_song01', 'sixth_S02']
SINGLES = ['seventh_06', 'seventh_07', 'seventh_08', 'seventh_09', 'seventh_10', 'seventh_11', 'sixth_S04', 'sixth_S05', 'sixth_S06', 'sixth_S07']


def cache(cfg):
    f = K.OUT/'fix_items_1p7.joblib'
    if f.exists(): return joblib.load(f)
    out = {}
    for k in K.TESTS+SONGS+SINGLES:
        raw = S.C['tests'][k]['raw'] if k in K.TESTS else S.C['sessions'][k]['raw']
        out[k] = items_at(raw, cfg, 1.7); print('items', k, len(out[k]), flush=True)
    joblib.dump(out, f); return out


def _offsets(X, p, tm, o, conf=.7):
    """Offset for hit i from confident hits j < i (cumulative per-drum sums); exact streaming equivalent."""
    n = len(X); pred = p.argmax(1); ok = p.max(1) >= conf
    sums = np.zeros((7, 21)); cnt = np.zeros(7); out = np.zeros((n, 21)); valid = np.zeros(n, bool)
    for i in range(n):
        m = cnt > 0
        if cnt[m].sum() >= o['minh'] and (cnt >= o['k_per_drum']).sum() >= o['min_drums']:
            v = (cnt[m, None]*tm[m]-sums[m]).sum(0)/cnt[m].sum(); nr = np.linalg.norm(v)
            if o['clamp'] and nr > o['clamp']: v = v*o['clamp']/nr
            out[i] = v; valid[i] = True
        if ok[i]: sums[pred[i]] += X[i, 83:104]; cnt[pred[i]] += 1
    return out, valid


def simulate(bundle, items, o):
    classes = bundle['classes']; tm = {h: np.array([bundle['drum_rotation_means'][f'{h}|{d}'] for d in classes]) for h in 'LR'}
    cym = set(o['cymbals']); P = []
    for h in 'LR':
        it = sorted([c for c in items if c['hand'] == h], key=lambda c: c['t'])
        if not it: continue
        X = np.stack([c['x'] for c in it]).astype(np.float32)
        p0 = R.probabilities(bundle, X, h)
        o1, v1 = _offsets(X, p0, tm[h], o); X1 = X.copy(); X1[:, 83:104] += o1
        p1 = np.where(v1[:, None], R.probabilities(bundle, X1, h), p0)
        o2, v2 = _offsets(X, p1, tm[h], o); X2 = X.copy(); X2[:, 83:104] += o2
        pf = np.where(v2[:, None], R.probabilities(bundle, X2, h), p1)
        strong_t = np.array([c['t'] for c in it if c['g'] >= 2.1])
        for c, q in zip(it, pf):
            d = classes[int(q.argmax())]
            if c['g'] < 2.1:
                if not (d in cym and q.max() >= o['score'] and c['g'] >= o['min_g']): continue
                if o['iso'] and len(strong_t) and np.min(np.abs(strong_t-c['t'])) < o['iso']: continue
            P.append(dict(t=c['t'], peak_ms=c['t'], hand=h, drum=d))
    return P


def evaluate(bundle, its, o):
    cells = defaultdict(Counter)
    for k in K.TESTS:
        P = simulate(bundle, its[k], o); truth = TRUTH[k]; pairs = match(truth, P)
        for i, g in enumerate(truth):
            if not g['hand']: continue
            j = pairs.get(i); cells[(g['hand'], g['drum'])]['ok' if j is not None and P[j]['hand'] == g['hand'] and P[j]['drum'] == g['drum'] else 'x'] += 1
    acc = {c: v['ok']/sum(v.values()) for c, v in cells.items()}
    song = Counter(); single = Counter()
    for k in SONGS+SINGLES:
        u, e = F.truth_of(k); res, _ = F.score(u, e, simulate(bundle, its[k], o))
        tgt = song if k in SONGS else single
        tgt['correct'] += res.get('correct', 0); tgt['midi'] += res['midi']; tgt['extra'] += res['extra']
    return acc, song, single


def main():
    bundle, cfg = core.load_components(); its = cache(cfg)
    base = dict(cymbals=['Hi-Hat', 'Crash', 'Ride'], score=.6, min_g=1.7, iso=0, minh=20, min_drums=1, k_per_drum=1, clamp=0)
    grid = [dict(base, name='目前第八次包')]
    grid += [dict(base, name='不救輕擊', min_g=2.1)]
    grid += [dict(base, name=f'救輕擊 分數≥{s}', score=s) for s in (.8, .9)]
    grid += [dict(base, name='只救 Crash/Ride', cymbals=['Crash', 'Ride'])]
    grid += [dict(base, name='救輕擊 ≥1.9g', min_g=1.9)]
    grid += [dict(base, name='救輕擊 遠離強峰120ms', iso=120)]
    grid += [dict(base, name=f'校正需≥{d}種鼓(各≥{k}下)', min_drums=d, k_per_drum=k) for d, k in ((2, 5), (3, 5), (2, 10))]
    grid += [dict(base, name=f'校正上限 {c}', clamp=c) for c in (.05, .1)]
    out = []
    for o in grid:
        acc, song, single = evaluate(bundle, its, o)
        rec = dict(name=o['name'], mean=round(float(np.mean(list(acc.values()))), 4), min=round(min(acc.values()), 4),
                   song_rate=round(song['correct']/song['midi'], 4), song_extra=song['extra'], single_rate=round(single['correct']/single['midi'], 4),
                   cells={f'{h}{d}': round(v, 3) for (h, d), v in acc.items()})
        out.append(rec); print(rec['name'], '| 14格平均', rec['mean'], '最低', rec['min'], '| 歌曲', rec['song_rate'], '多判', rec['song_extra'], '| 單手', rec['single_rate'], flush=True)
        (K.OUT/'fix_sim.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
