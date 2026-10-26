"""Parallel field-engine replay of every round 4-7 recording (8th package), scored like full_report,
plus pretest 14 cells. Output: out/scorecard_hitgate.json"""
import json, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np, joblib

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
PACK = ROOT/'第八次的包_暫定2026-10-27'


def work(args):
    name, raw = args
    sys.path[:0] = [str(PACK/'Program'), str(HERE)]
    import seventh_core as core
    from verify_1023 import stream
    bundle, config = core.load_components()
    P, eng = stream(bundle, config, raw)
    return name, [dict(t=p['peak_ms'], peak_ms=p['peak_ms'], hand=p['hand'], drum=p['drum']) for p in P], dict(eng.runtime.rejected)


def main():
    sys.path[:0] = [str(PACK/'Program'), str(HERE)]
    import seventh_core  # noqa: F401  (8th package modules first)
    import full_report as F, common as K
    from verify_1023 import TRUTH, match, C
    jobs = [(n, C['tests'][n]['raw'] if n in K.TESTS else C['sessions'][n]['raw']) for n, _, _ in F.ORDER]
    preds = {}
    with Pool(12) as pool:
        for name, P, rej in pool.imap_unordered(work, jobs): preds[name] = (P, rej); print('done', name, len(P), rej, flush=True)
    rows = []; cells = defaultdict(Counter)
    for name, title, role in F.ORDER:
        P, rej = preds[name]; used, excl = F.truth_of(name); res, _ = F.score(used, excl, P)
        res.update(name=name, title=title, role=role, rejected=rej); rows.append(res)
        if name in K.TESTS:
            truth = TRUTH[name]; pairs = match(truth, P)
            for i, g in enumerate(truth):
                if not g['hand']: continue
                j = pairs.get(i); o = 'missed' if j is None else ('wrong_hand' if P[j]['hand'] != g['hand'] else ('ok' if P[j]['drum'] == g['drum'] else P[j]['drum']))
                cells[(g['hand'], g['drum'])][o] += 1
    cell_rows = [dict(hand=h, drum=d, n=sum(cells[(h, d)].values()), ok=cells[(h, d)]['ok'], acc=round(cells[(h, d)]['ok']/max(sum(cells[(h, d)].values()), 1), 4),
                      top=Counter({k: v for k, v in cells[(h, d)].items() if k != 'ok'}).most_common(2)) for h in 'LR' for d in K.DRUMS]
    (K.OUT/'scorecard_hitgate.json').write_text(json.dumps(dict(recordings=rows, cells_pretest=cell_rows), ensure_ascii=False, indent=1, default=int), encoding='utf-8')
    for r in rows: print(r['title'], {k: r.get(k, 0) for k in ('midi', 'detected', 'correct', 'wrong_drum', 'missed', 'extra')})
    print('cells', ' '.join(f"{c['hand']}{c['drum']}{c['acc']*100:.0f}" for c in cell_rows), 'mean', round(np.mean([c['acc'] for c in cell_rows])*100, 1))


if __name__ == '__main__':
    main()
