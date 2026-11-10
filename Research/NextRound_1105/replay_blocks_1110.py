"""11/10 pretests without MIDI: answer key from the played protocol (block order told by the user, block
boundaries = the order of that hand's live hits; pauses between blocks were checked by hand), then causal
replay through the field engine with switches, to see which engine part causes whole-block errors.

  python replay_blocks_1110.py [--model out/model_x.joblib] [--variants base,noadapt,...]
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('--code', default='11-10的包'); ap.add_argument('--model', default='out/model_r10c.joblib')
ap.add_argument('--variants', default='base,noadapt,noprior,nocont,noanchor,nozero'); ap.add_argument('--tag', default='r10c')
ARGS = ap.parse_args() if __name__ == '__main__' else ap.parse_args([a for a in sys.argv[1:]] if any(a.startswith('--') for a in sys.argv[1:]) else [])
import joblib
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/ARGS.code/'Program'), str(HERE)]
import seventh_core as core
S, H, C, T1, T2, R, F = '小鼓', 'Hi-Hat', 'Crash', '高音 Tom', '中音 Tom', 'Ride', '落地 Tom'
FWD = [(S, 10), (S, 20), (H, 20), (C, 20), (T1, 20), (T2, 20), (R, 20), (F, 20)]
KEYS = {
 '133616 正打': dict(L=[(S, 10), (S, 20), (None, 1), (H, 20), (C, 20), (T1, 20), (T2, 20), (R, 20), (F, 20)], R=FWD),
 '134421 反打': dict(L=[(S, 10), (F, 20), (R, 18), (T2, 20), (T1, 20), (C, 20), (H, 20), (S, 20)], R=[(S, 10), (F, 12), (R, 20), (T2, 20), (T1, 20), (C, 20), (H, 20), (S, 20)]),
 '133951 高中(沒開局)': dict(L=[(T1, 20), (T2, 20)], R=[(T1, 20), (T2, 20)]),
 '134211 高中(有開局)': dict(L=[(S, 10), (T1, 20), (T2, 20)], R=[(S, 10), (T1, 20), (T2, 20)]),
}
MD = ROOT/ARGS.code/'Data/模型實測'


def truth_of(key):
    D = next(p for p in MD.glob('*') if key.split()[0] in p.name)
    P = sorted((r for r in csv.DictReader(open(D/'predictions.csv', encoding='utf-8-sig')) if r['accepted'] in ('True', '1')), key=lambda r: float(r['peak_ms']))
    out = []
    for h in 'LR':
        Q = [p for p in P if p['hand'] == h]; seq = [d for d, n in KEYS[key][h] for _ in range(n)]
        assert len(seq) == len(Q), (key, h, len(seq), len(Q))
        out += [dict(hand=h, t=float(p['peak_ms']), drum=d, live=p['drum']) for p, d in zip(Q, seq) if d]
    return next(D.glob('*_raw_100hz_*.csv')), out


def work(job):
    key, variant, model = job
    raw, truth = truth_of(key)
    b = joblib.load(model); core.runtime.attach_boosters(b); config = core.load_components()[1]
    rt = core.runtime; anchor = 5
    if variant == 'noadapt': b['adaptation'] = dict(b['adaptation'], alpha=0.0)
    if variant == 'noprior': b['position_prior'] = None
    if variant == 'nocont': rt.CONTINUITY = dict(rt.CONTINUITY, below_score=0.0)
    if variant == 'noanchor': anchor = 0
    if variant == 'nozero': b.pop('_zero_estimator', None)
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(b, config, snare_anchor=anchor); P = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]
        p, _ = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        P.extend(p)
    cells = defaultdict(lambda: [0, 0, Counter()]); live_ok = 0
    for g in truth:
        near = [p for p in P if p['hand'] == g['hand'] and abs(p['peak_ms']-g['t']) <= 30]
        d = near[0]['drum'] if near else 'miss'; c = cells[f"{g['hand']}|{g['drum']}"]; c[1] += 1
        if d == g['drum']: c[0] += 1
        else: c[2][d] += 1
        live_ok += g['live'] == g['drum']
    r = engine.runtime
    return key, variant, dict(ok=sum(c[0] for c in cells.values()), n=len(truth), live_ok=live_ok, anchored={h: (r.classes[r.anchored_drum[h]] if r.anchored_drum[h] is not None else None) for h in 'LR'},
                              zero_rot=dict(r.zero_rot), cells={k: [v[0], v[1], dict(v[2])] for k, v in sorted(cells.items())})


if __name__ == '__main__':
    V = ARGS.variants.split(','); J = [(k, v, str(HERE/ARGS.model)) for k in KEYS for v in V]; res = defaultdict(dict)
    with Pool(6) as pool:
        for key, variant, r in pool.imap_unordered(work, J, chunksize=1): res[key][variant] = r
    for key in KEYS:
        print('==', key, 'live', res[key][V[0]]['live_ok'], '/', res[key][V[0]]['n'])
        for v in V:
            r = res[key][v]; bad = {k: f"{c[0]}/{c[1]} {c[2]}" for k, c in r['cells'].items() if c[0] < 0.85*c[1]}
            print(f"   {v:9s} {r['ok']:3d}/{r['n']}  anchor {r['anchored']} zero {r['zero_rot']}  bad: {bad}")
    (HERE/f'out/replay_blocks_{ARGS.tag}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
