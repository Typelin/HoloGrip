"""Sample-by-sample replay of selected non-pretest recordings through the 9th package field engine
(anchor mode as in the field), scored against their labels; reports zero-guard rebuilds and timing."""
import sys, time
import numpy as np
sys.argv = ['x']
import zero_tolerance_9th as Z
import eval_all_9th as A
core = Z.core


def main(names):
    rows, names_all, Lb, T8, R8 = A.FB.setup(); bundle, config = core.load_components()
    for nm in names:
        raw = A.L5.r8_docs()[nm]['raw_csv'] if nm.startswith('eighth_') else Z.V9.V.C['sessions'][nm]['raw']
        t0 = time.perf_counter(); P = Z.stream_zero_shift(bundle, config, raw, 0.0); dt = time.perf_counter()-t0
        for p in P: p['t'] = float(p['peak_ms'])
        used, excl = T8[nm] if nm in T8 else A.F.truth_of(nm); r, _ = A.F.score(used, excl, P)
        corr = sorted({(p['hand'], p.get('zero_correction_deg', 0.0)) for p in P if p.get('zero_correction_deg')})
        print(nm, f"{r.get('correct', 0)}/{r['midi']}", '歸零修正', corr or '無', f'重播 {dt:.0f} s', flush=True)


if __name__ == '__main__':
    main(['eighth_fioor ton L', 'eighth_ton2 L', 'sixth_S02'])
