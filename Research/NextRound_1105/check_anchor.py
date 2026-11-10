"""All pretests (full + LODO bundles) with the pretest snare anchor forced: the first N hits of each hand are
taken as snare (the pretest score always opens with 10 snare hits per hand) instead of only when the committee agrees."""
import json, sys
import score_params as SP
core, RD = SP.core, SP.RD
def main():
    RD.setup(); _, config = core.load_components(); raws = SP.raw_paths()
    full = SP.load(SP.HERE/'out/model_ex.joblib'); lodo = {D: SP.load(RD.OUT/f'model_lodoex_{D}.joblib') for D in SP.FB.DAYS}
    rt = core.runtime; saved = dict(rt.ANCHOR_RULE); n0 = rt.PRETEST_SNARE_ANCHOR; res = {}
    for name, n, force in [('base', n0, False), ('force5', 5, True), ('force10', 10, True)]:
        rt.ANCHOR_RULE.clear(); rt.ANCHOR_RULE.update(saved)
        if force: rt.ANCHOR_RULE.update(snare_in_top=99, max_pose_deg=1e9)
        rt.PRETEST_SNARE_ANCHOR = n
        r = dict(full=SP.run({D: full for D in SP.FB.DAYS}, config, SP.BASE, raws), lodo=SP.run(lodo, config, SP.BASE, raws)); res[name] = r
        print(name, 'full', r['full']['ok'], 'fail', r['full']['failed'], '| lodo', r['lodo']['ok'], 'fail', r['lodo']['failed'], flush=True)
        for k in ('full', 'lodo'):
            print('  ', k, ' '.join(f"{t.replace('pretest:','')}={v['ok']}/{v['n']}" for t, v in r[k]['takes'].items()), flush=True)
    rt.ANCHOR_RULE.clear(); rt.ANCHOR_RULE.update(saved); rt.PRETEST_SNARE_ANCHOR = n0
    (RD.OUT/'score_anchor_force.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
if __name__ == '__main__':
    main()
