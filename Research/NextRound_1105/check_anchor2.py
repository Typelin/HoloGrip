"""Anchor-rule variants on every pretest (full + LODO bundles) and on today's pretest (provisional block score)."""
import csv, json, sys
from collections import Counter
from pathlib import Path
import score_params as SP
core, RD = SP.core, SP.RD
TODAY = Path('C:/Users/Typelin_Station/Desktop/HoloGrip/第十次的包_暫定/Data/模型實測/S9_P01_sevendrum_take01_20261106_103332_121711/S9_P01_sevendrum_take01_raw_100hz_20261106_103332.csv')
ORDER = ['小鼓', '小鼓', 'Hi-Hat', 'Crash', '高音 Tom', '中音 Tom', 'Ride', '落地 Tom']
VARIANTS = [('base', {}), ('top2', dict(snare_in_top=2)), ('top3', dict(snare_in_top=3)),
            ('tight8_any', dict(snare_in_top=99)), ('tight12_any', dict(snare_in_top=99, max_pose_deg=12.0)),
            ('tight12_top3', dict(snare_in_top=3, max_pose_deg=12.0))]
def today(b, config):
    P = SP.stream(b, config, str(TODAY)); ok = n = 0; cells = {}
    for h in 'LR':
        a = sorted([p for p in P if p['hand'] == h], key=lambda p: float(p['peak_ms'])); k = 0
        for i, p in enumerate(a):
            if i and float(p['peak_ms'])-float(a[i-1]['peak_ms']) > 3000: k += 1
            d = ORDER[min(k, 7)]; c = cells.setdefault(f'{h}|{d}', [0, 0]); c[0] += p['drum'] == d; c[1] += 1
    ok = sum(c[0] for c in cells.values()); n = sum(c[1] for c in cells.values())
    return ok, n, sum(c[0] < 0.85*c[1] for c in cells.values()), cells
def main():
    RD.setup(); _, config = core.load_components(); raws = SP.raw_paths()
    full = SP.load(SP.HERE/'out/model_ex.joblib'); lodo = {D: SP.load(RD.OUT/f'model_lodoex_{D}.joblib') for D in SP.FB.DAYS}
    rt = core.runtime; saved = dict(rt.ANCHOR_RULE); res = {}
    for name, over in VARIANTS:
        rt.ANCHOR_RULE.clear(); rt.ANCHOR_RULE.update(saved, **over)
        F = SP.run({D: full for D in SP.FB.DAYS}, config, SP.BASE, raws); Lo = SP.run(lodo, config, SP.BASE, raws)
        core.runtime.attach_boosters(full); t = today(full, config)
        res[name] = dict(full=F, lodo=Lo, today=t)
        print(name, f"full {F['ok']} fail {F['failed']} | lodo {Lo['ok']} fail {Lo['failed']} | today {t[0]}/{t[1]} fail {t[2]}", flush=True)
        for k, r in (('full', F), ('lodo', Lo)):
            print('  ', k, ' '.join(f"{x.replace('pretest:','')}={v['ok']}" for x, v in r['takes'].items()), flush=True)
        print('   today', {k: f'{v[0]}/{v[1]}' for k, v in t[3].items() if v[0] < v[1]}, flush=True)
    rt.ANCHOR_RULE.clear(); rt.ANCHOR_RULE.update(saved)
    (RD.OUT/'score_anchor2.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
if __name__ == '__main__':
    main()
