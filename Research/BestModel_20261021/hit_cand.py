"""Hit-detection study, step 1: low-threshold candidates (1.5 g) for every recording, in parallel.
Per candidate: hand, peak time, peak g, the detector's own causal shape features, the 104 venue-free
features, plus context: other hand's max |a| within +-40 ms, time/g of the previous candidate of the
same hand. All causal (window up to peak + 80 ms)."""
import sys, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np, joblib

ROOT = Path(__file__).resolve().parents[2]
P7 = ROOT/'第七次的包_暫定2026-10-20'
OUT = Path(__file__).resolve().parent/'out'


def work(args):
    name, raw, thr, extra = args
    sys.path[:0] = [str(P7/'Program'), str(ROOT/'第六次的包_暫定2026-10-16/Program')]
    import venue_free_runtime_1020 as rt, causal_hit_detector_1016 as det, json as js
    cfg = js.loads((P7/'Model/detector_venue_free_2p1g.json').read_text(encoding='utf-8'))['config']
    hands = rt.world.read_raw(raw); prep = rt.world.base.legacy.prepare(hands)
    peaks, _ = det.detect(hands, dict(cfg, threshold_g=thr, **extra)); out = []
    ser = {h: (np.array([r['t'] for r in a]), np.array([r['mag'] for r in a])) for h, a in hands.items()}
    last = {}
    for p in sorted(peaks, key=lambda p: p['peak_ms']):
        s = prep[p['hand']]['rows'][p['available_index']]
        x = rt.feature(prep, p, (s['absolute_yaw']-s['yaw']+180) % 360-180, s['absolute_pitch']-s['pitch'])
        if x is None: continue
        oh = 'R' if p['hand'] == 'L' else 'L'; ts, mg = ser[oh]; i, j = np.searchsorted(ts, p['peak_ms']-40), np.searchsorted(ts, p['peak_ms']+40, 'right')
        other = float(mg[i:j].max()) if j > i else 1.0
        pv = last.get(p['hand'])
        ctx = [other, np.log1p(min(p['peak_ms']-pv[0], 3000)) if pv else np.log1p(3000), p['peak_g']/pv[1] if pv else 1.0]
        last[p['hand']] = (p['peak_ms'], p['peak_g'])
        out.append(dict(hand=p['hand'], t=float(p['peak_ms']), g=float(p['peak_g']), det=np.asarray(p.get('feature', []), np.float32),
                        x=x.astype(np.float32), ctx=np.asarray(ctx, np.float32)))
    return name, out


def main():
    sys.path[:0] = [str(Path(__file__).resolve().parent)]
    C = joblib.load(P7/'Data/實驗_姿態模型自動標記_20261020/實驗快取.joblib')
    tag = sys.argv[1] if len(sys.argv) > 1 else '1p5'
    thr, extra = {'1p5': (1.5, {}), 'loose': (1.3, dict(rise_g=0.4, refractory_ms=60.0, merge_ms=60.0)), 'thr13': (1.3, {}), 'ref60': (1.5, dict(refractory_ms=60.0, merge_ms=60.0))}[tag]
    jobs = [(k, v['raw'], thr, extra) for k, v in C['tests'].items()]+[(k, v['raw'], thr, extra) for k, v in C['sessions'].items()]
    res = {}
    with Pool(12) as pool:
        for name, out in pool.imap_unordered(work, jobs):
            res[name] = out; print(name, len(out), flush=True)
    joblib.dump(res, OUT/f'hit_cand_{tag}.joblib')


if __name__ == '__main__':
    main()
