"""Create `11-24的包` from `11-17的包` (same program and runtime: position-prior floor 0.1, fast 'start', dated session names)
with a model retrained on 10/09-11/17 (R4..R14; R14 = 11/17 人員3 song 1 + left single-hand drill; the right drill is left out,
see 現場紀錄_20261117_ZH_TW.md). Field data is never copied: Data/, QA/ and 今天/ start empty.
Refuses to touch a destination that holds field data.

  HOLOGRIP_ROUNDS=out/rounds_R14.json python make_package_1124.py --model=out/model_h5.joblib [--replace]
"""
from __future__ import annotations
import json, re, shutil, sys
from pathlib import Path
import joblib
import make_package_1112 as P   # helpers: sha256, run, DATA_DIRS, ENV
HERE = P.HERE; ROOT = P.ROOT
SRC = ROOT/'11-17的包'; DST = ROOT/'11-24的包'; TAG = '1124'; DATE = '2026-11-24'
MODEL = HERE/next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--model=')), 'out/model_h5.joblib')


def patch(rel, pairs):
    p = DST/rel; s = p.read_bytes().decode('utf-8')
    for old, new in pairs:
        n = s.count(old); assert n == 1, f'{rel}: expected 1 occurrence of {old!r}, found {n}'
        s = s.replace(old, new)
    p.write_bytes(s.encode('utf-8'))


def main():
    if DST.exists():
        held = [p for d in ('Data', '今天') for p in (DST/d).rglob('*') if p.is_file()]
        assert not held, f'{DST} already holds field data ({len(held)} files); refusing to touch it'
        assert '--replace' in sys.argv, f'{DST} exists; pass --replace to rebuild it'
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns('__pycache__', 'Data', 'QA', '今天', 'package_manifest.json'))
    for d in P.DATA_DIRS: (DST/'Data'/d).mkdir(parents=True, exist_ok=True)
    (DST/'QA').mkdir(exist_ok=True); (DST/'今天').mkdir(exist_ok=True)
    shutil.copyfile(MODEL, DST/'Model/venue_free_committee_1023.joblib')
    b = joblib.load(MODEL); digest = P.sha256(DST/'Model/venue_free_committee_1023.joblib'); n = b['training_events']
    S = json.loads(MODEL.with_suffix('.json').read_text(encoding='utf-8')); days = ','.join(S['days'])
    core = (SRC/'Program/seventh_core.py').read_text(encoding='utf-8')
    old = re.search(r"MODEL_SHA256 = '([0-9a-f]{64})'", core).group(1); n0 = int(re.search(r'TRAINING_EVENTS = (\d+)', core).group(1))
    k0 = int(re.search(r"len\(bundle\['portable'\]\) != (\d+)", core).group(1)); k = len(b['portable'])
    frz = (SRC/'Program/freeze_seventh_package.py').read_text(encoding='utf-8')
    td = re.search(r"target_date='[^']*',prepared_date='[^']*'", frz).group(0); wr = re.search(r"workflow_revision='[^']*'", frz).group(0)
    ts = re.search(r"training_scope='[^ ]*", frz).group(0)
    patch('Program/seventh_core.py', [(old, digest), (f'TRAINING_EVENTS = {n0}', f'TRAINING_EVENTS = {n}'), ("MODEL_NAME = '10/09～11/12 ", "MODEL_NAME = '10/09～11/17 ")])
    patch('Program/freeze_seventh_package.py', [(old, digest), (f"bundle['training_events']=={n0}", f"bundle['training_events']=={n}"), (f'training_events={n0}', f'training_events={n}'),
                                                (td, f"target_date='{DATE}',prepared_date='{S['built']}'"),
                                                (wr, f"workflow_revision='{DATE}-pkg{TAG}-{'-'.join(S['days'])}-retrained-ninth-recipe'"),
                                                (ts, f"training_scope='{days}")])
    patch('Program/test_seventh_package.py', [(f"self.assertEqual(self.bundle['training_events'],{n0})", f"self.assertEqual(self.bundle['training_events'],{n})"),
                                              ("out/replay_1117'", f"out/replay_{TAG}'")])
    patch('Program/seventh_collect.py', [('HoloGrip｜11/17 的包｜', 'HoloGrip｜11/24 的包｜')])
    patch('Program/seventh_native_gui.py', [('HoloGrip｜11/17 的包｜', 'HoloGrip｜11/24 的包｜'),
                                            ('11/17 的包｜打擊判斷＋10/09～11/12 資料訓練的模型', '11/24 的包｜打擊判斷＋10/09～11/17 資料訓練的模型')])
    if k != k0:
        patch('Program/seventh_core.py', [(f"len(bundle['portable']) != {k0}", f"len(bundle['portable']) != {k}")])
        patch('Program/freeze_seventh_package.py', [(f"assert len(bundle['portable'])=={k0}", f"assert len(bundle['portable'])=={k}")])
        patch('Program/test_seventh_package.py', [(f"self.assertEqual(len(self.bundle['portable']),{k0})", f"self.assertEqual(len(self.bundle['portable']),{k})")])
    py = sys.executable; rep = dict(src=SRC.name, dst=DST.name, model=str(MODEL), model_sha256=digest, training_events=n, days=S['days'], excluded=S['excluded'], steps={})
    rep['steps']['freeze'] = P.run([py, 'Program/freeze_seventh_package.py'], DST, f'log_{TAG}_freeze.log')
    rep['steps']['references'] = P.run([py, str(HERE/'score.py'), '--code', DST.name, '--days', 'R6,R7', '--tag', f'refs_{TAG}', '--refs', f'replay_{TAG}'], HERE, f'log_{TAG}_refs.log')
    rep['steps']['package_tests'] = P.run([py, 'Program/test_seventh_package.py'], DST, f'log_{TAG}_tests.log')
    rep['steps']['gui_test'] = P.run([py, 'Program/test_seventh_native_gui.py'], DST, f'log_{TAG}_gui.log')
    rep['steps']['self_check'] = P.run([py, 'Program/seventh_check.py'], DST, f'log_{TAG}_check.log')
    rep['passed'] = all(v[0] == 0 for v in rep['steps'].values())
    (HERE/'out'/f'package_{TAG}.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    for key, (code, tail) in rep['steps'].items(): print(key, 'exit', code, '|', tail.strip().splitlines()[-1][:200] if tail.strip() else '', flush=True)
    print('PASSED' if rep['passed'] else 'FAILED', DST, digest[:12], n, flush=True)


if __name__ == '__main__':
    main()
