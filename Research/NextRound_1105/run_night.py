"""One command for the night after a field day (after rounds.json has been filled and checked).

  python run_night.py R9 --name 第十次的包_暫定 --ordinal 第十次 --date 2026-11-06 [--lodo]

1. autolabel_day.py R9                     label the day's songs / four-drum / single-hand takes
2. score.py --live <package> --days R9     blind result: what the field program really answered today
3. score.py --days R9                      same pretest replayed through the package (must equal step 2's engine)
4. build_model.py --tag <tag>              retrain with the new day (pretest excluded)
5. make_package.py ...                     new package: install, freeze, references, tests, GUI test
6. score.py --code <new package>           development score of the new package on every pretest
7. (--lodo) build_model.py --hold D for every day + score.py --lodo   honest leave-one-day-out
Each step's output is in out/log_night_<R>_<step>.log; the summary in out/night_<R>.json and .md.
Stops at the first failing step.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('day')
ap.add_argument('--name', required=True); ap.add_argument('--ordinal', required=True); ap.add_argument('--date', required=True)
ap.add_argument('--lodo', action='store_true')
ap.add_argument('--skip', default='', help='comma list of step numbers to skip (e.g. 1 when labels were already made)')
A = ap.parse_args()

HERE = Path(__file__).resolve().parent; OUT = HERE/'out'; OUT.mkdir(exist_ok=True)
ENV = dict(os.environ, PYTHONIOENCODING='utf-8'); PY = sys.executable


def step(no, name, args):
    if str(no) in A.skip.split(','): print(no, name, 'skipped', flush=True); return
    t = time.time(); log = OUT/f'log_night_{A.day}_{no}_{name}.log'
    with open(log, 'w', encoding='utf-8') as f:
        code = subprocess.run([PY, *args], cwd=HERE, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode
    print(no, name, 'exit', code, f'{(time.time()-t)/60:.1f} min', log.name, flush=True)
    if code: sys.exit(f'step {no} {name} failed; see {log}')


def main():
    cfg = json.loads((HERE/os.environ.get('HOLOGRIP_ROUNDS', 'rounds.json')).read_text(encoding='utf-8'))[A.day]
    assert cfg.get('enabled'), f'{A.day} is not enabled in rounds.json'
    tag = A.name.split('的包')[0]
    step(1, 'label', ['autolabel_day.py', A.day])
    step(2, 'live', ['score.py', '--live', cfg['package'], '--days', A.day, '--tag', f'live_{A.day}'])
    step(3, 'replay', ['score.py', '--code', cfg['package'], '--days', A.day, '--tag', f'replay_{A.day}'])
    step(4, 'build', ['build_model.py', '--tag', tag])
    step(5, 'package', ['make_package.py', '--model', f'out/model_{tag}.joblib', '--name', A.name, '--ordinal', A.ordinal, '--date', A.date])
    step(6, 'full', ['score.py', '--code', A.name, '--tag', f'full_{tag}'])
    if A.lodo:
        import rounds_data as RD
        days = ['R4', 'R5', 'R6', 'R7', 'R8']+list(RD.config())
        for D in days: step(7, f'lodo_{D}', ['build_model.py', '--tag', f'lodo_{D}', '--hold', D])
        step(8, 'lodo_score', ['score.py', '--lodo', '--tag', f'lodo_{tag}'])
    md = [f'# {A.day} 當晚結果\n']
    for t, title in [(f'live_{A.day}', '今天現場實際判斷（盲測）'), (f'replay_{A.day}', '同一份前測重播'),
                     (f'full_{tag}', f'{A.ordinal}包，全部訓練（開發分數）'), (f'lodo_{tag}', '留一天（誠實跨天）')]:
        p = OUT/f'score_{t}.md'
        if p.exists(): md += [f'## {title}\n', p.read_text(encoding='utf-8'), '']
    pk = json.loads((OUT/f'package_{tag}.json').read_text(encoding='utf-8'))
    md += [f"## 新包\n`{A.name}`：模型 `{pk['model_sha256'][:12]}…`，{pk['training_events']:,} 下，測試 {'全部通過' if pk['passed'] else '失敗'}。\n"]
    (OUT/f'night_{A.day}.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md), flush=True)


if __name__ == '__main__':
    main()
