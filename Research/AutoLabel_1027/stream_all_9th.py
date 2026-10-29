"""Sample-by-sample replay of ALL round 4-8 recordings through the 9th package field engine (pretests with the
cleaned answer key, other recordings against their labels). Field engine = what runs live."""
import sys, json
ONLY = sys.argv[1] if len(sys.argv) > 1 else ''   # read before imports that rewrite sys.argv
import numpy as np
import zero_tolerance_9th as Z
import eval_all_9th as A
core = Z.core


def main(only=None):
    rows, names_all, Lb, T8, R8 = A.FB.setup(); bundle, config = core.load_components(); docs = A.L5.r8_docs(); res = {}
    for nm in [n for n in sorted(A.H.CAND, key=lambda n: (A.L5.day_of(n), n)) if not only or n in only or A.L5.day_of(n) in only]:
        if nm == A.L5.PRE: raw = str(A.L5.PRE_RAW)
        elif nm.startswith('eighth_'): raw = docs[nm]['raw_csv']
        elif nm in A.K.TESTS: raw = Z.V9.V.C['tests'][nm]['raw']
        else: raw = Z.V9.V.C['sessions'][nm]['raw']
        P = Z.stream_zero_shift(bundle, config, raw, 0.0)
        for p in P: p['t'] = float(p['peak_ms'])
        if nm in A.K.TESTS:
            T = A.TC.dedupe(A.L5.TRUTH[nm], A.H.CAND[nm])[0]; pr = A.L5.match(T, P)
            ok = sum(1 for i, g in enumerate(T) if (j := pr.get(i)) is not None and (g['hand'] is None or P[j]['hand'] == g['hand']) and P[j]['drum'] == g['drum'])
            res[nm] = ('前測', ok, len(T))
        else:
            used, excl = T8[nm] if nm in T8 else A.F.truth_of(nm); r, _ = A.F.score(used, excl, P); res[nm] = ('訓練', r.get('correct', 0), r['midi'])
        corr = sorted({(p['hand'], p['zero_correction_deg']) for p in P if p.get('zero_correction_deg')})
        print(A.L5.day_of(nm), nm, f'{res[nm][1]}/{res[nm][2]}', corr or '', flush=True)
    for k in ['前測', '訓練']:
        a = sum(v[1] for v in res.values() if v[0] == k); b = sum(v[2] for v in res.values() if v[0] == k); print(k, f'{a}/{b} {a/b*100:.1f}%')
    a = sum(v[1] for v in res.values()); b = sum(v[2] for v in res.values()); print('全部', f'{a}/{b} {a/b*100:.1f}%')
    (Z.HERE/'out/stream_all_9th.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(set(ONLY.split(',')) if ONLY else None)
