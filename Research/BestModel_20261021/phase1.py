"""Phase 1 grid: data mix, weighting, learners, capacity. Results appended to out/phase1.jsonl."""
import json, time, sys
import common as K

CONFIGS = {
    'base_ens': {},
    'xgb_only': dict(mlp=False),
    'mlp_only': dict(xgb=False),
    'no_class_bal': dict(class_bal='none'),
    'group_bal': dict(group_bal=True),
    'r7_x2': dict(group_w={'R7o': 2., 'R7m': 2.}),
    'r7m_x3': dict(group_w={'R7m': 3.}),
    'min_g_2.6': dict(min_g=2.6),
    'xgb_deep': dict(xgb_params=dict(max_depth=8, n_estimators=1000, learning_rate=.03)),
    'xgb_shallow': dict(xgb_params=dict(max_depth=4, n_estimators=1500, learning_rate=.03)),
    'mlp_wide': dict(mlp_params=dict(hidden=(1024, 512, 256), drop=.3)),
    'mlp_long': dict(mlp_params=dict(epochs=250, drop=.3)),
    'mlp_reg': dict(mlp_params=dict(drop=.4, wd=1e-2, smooth=.1)),
}


def main():
    D = K.build(); rows = D['rows']
    print('rows', len(rows), flush=True)
    done = set()
    f = K.OUT/'phase1.jsonl'
    if f.exists(): done = {json.loads(l)['name'] for l in f.read_text(encoding='utf-8').splitlines() if l.strip()}
    for name, cfg in CONFIGS.items():
        if name in done: continue
        t0 = time.time()
        com = K.train(rows, cfg, seeds=(0, 1)); res, summ = K.exam(com, D)
        cv = K.group_cv(rows, cfg)
        rec = dict(name=name, cfg=cfg, pretest=summ, per={k: v['0.0'] for k, v in res.items()}, cv=cv, sec=round(time.time()-t0))
        with f.open('a', encoding='utf-8') as fh: fh.write(json.dumps(rec, ensure_ascii=False, default=float)+'\n')
        print(name, summ['0.0'], summ['0.6'], 'cv', cv['mean_acc'], {g: cv[g]['acc'] for g in ['R4', 'R5', 'R6', 'R7o', 'R7m']}, rec['sec'], 's', flush=True)


if __name__ == '__main__':
    main()
