"""Does pose continuity carry the previous drum block into the next one (pretest drum changes)?
Replays the five pretests through the installed field engine with continuity variants (runtime constant
patched in memory only) and reports totals plus every cell below 17/20."""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_package_1027 as VP
V, FB, L5, core = VP.V, VP.FB, VP.L5, VP.core

VARIANTS = {
    'off': dict(max_ms=600.0, max_deg=-1.0, below_score=0.9, previous_in_top=2),
    'deg8': dict(max_ms=600.0, max_deg=8.0, below_score=0.9, previous_in_top=2),
    'score0.6': dict(max_ms=600.0, max_deg=16.0, below_score=0.6, previous_in_top=2),
}


def main():
    FB.setup(); bundle, config = core.load_components()
    raws = {k: V.C['tests'][k]['raw'] for k in V.TESTS}; raws[L5.PRE] = str(L5.PRE_RAW)
    out = {}
    for name, cont in VARIANTS.items():
        core.runtime.CONTINUITY.clear(); core.runtime.CONTINUITY.update(cont)
        tot = [0, 0]; rep = {}
        for D, names in FB.PRETESTS.items():
            ok = n = 0; cells = {}
            for nm in names:
                P, _ = V.stream(bundle, config, raws[nm], anchor=core.runtime.PRETEST_SNARE_ANCHOR)
                o, m, c = VP.score(L5.TRUTH[nm], P); ok += o; n += m
                for k, v in c.items():
                    cc = cells.setdefault(k, [0, 0, {}]); cc[0] += v[0]; cc[1] += v[1]
                    for w, q in v[2].items(): cc[2][w] = cc[2].get(w, 0)+q
            tot[0] += ok; tot[1] += n
            rep[D] = dict(ok=ok, n=n, rate=round(ok/n, 4), below={k: f'{v[0]}/{v[1]} {v[2]}' for k, v in sorted(cells.items()) if v[0]/v[1] < .85})
            print(name, D, ok, '/', n, round(ok/n, 4), rep[D]['below'], flush=True)
        out[name] = dict(total=tot, days=rep); print(name, '合計', tot, round(tot[0]/tot[1], 4), flush=True)
        (HERE/'out/continuity_test_1027.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
