"""Data cleaning: baseline (all training recordings except held-out song 04), then drop each training
recording once. Score = pretest 14 cells (min, mean) + held-out song accuracy. 2 seeds."""
import json
import numpy as np
import common as K, cells as Q, h_common as H


def run(train, hold, tests, seeds=(0, 1)):
    com = K.train(train, {}, seeds=seeds); r = H.pretest(com, tests); s = H.song_acc(com, hold)
    return dict(min=r['min'], mean=r['mean'], overall=r['overall'], song=s, acc={f'{a}{b}': v for (a, b), v in r['acc'].items()})


def main():
    train, hold, tests = H.load(with_timing=False); out = {}
    out['ALL'] = run(train, hold, tests); print('ALL', out['ALL']['min'], out['ALL']['mean'], out['ALL']['song'], flush=True)
    for s in sorted({r['session'] for r in train}):
        out[s] = run([r for r in train if r['session'] != s], hold, tests)
        print('drop', s, round(out[s]['min'], 3), round(out[s]['mean'], 4), round(out[s]['song'], 4), flush=True)
        (K.OUT/'h1_drop_one.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
