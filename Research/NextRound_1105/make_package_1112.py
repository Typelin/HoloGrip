"""Create `11-12的包` from `11-10的包` (same program: wired auto-reopen, wireless, live judgement while collecting,
relaxed snare anchor) with a model retrained on 10/09-11/10 (--model=out/model_<tag>.joblib; default c12only11, shipped: e1).
Only the frozen-model identity, window titles, manifest dates and the test reference folder change.
Field data is never copied: Data/, QA/ and 今天/ start empty. Refuses to touch a destination that holds field data.

  HOLOGRIP_ROUNDS=out/rounds_1112.json python make_package_1112.py [--replace]
"""
from __future__ import annotations
import hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
import joblib
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
SRC = ROOT/'11-10的包'; DST = ROOT/'11-12的包'; MODEL = HERE/next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--model=')), 'out/model_c12only11.joblib'); TAG = '1112'; DATE = '2026-11-12'
DATA_DIRS = ['MIDI', 'Raw', '全程監聽', '對齊與異常報告', '模型實測']
ENV = dict(os.environ, PYTHONIOENCODING='utf-8')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def patch(rel, pairs):
    p = DST/rel; s = p.read_bytes().decode('utf-8')
    for old, new in pairs:
        n = s.count(old); assert n == 1, f'{rel}: expected 1 occurrence of {old!r}, found {n}'
        s = s.replace(old, new)
    p.write_bytes(s.encode('utf-8'))


def run(cmd, cwd, log):
    r = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, encoding='utf-8', errors='replace')
    (HERE/'out'/log).write_text(r.stdout+'\n--- stderr ---\n'+r.stderr, encoding='utf-8')
    return r.returncode, (r.stdout+r.stderr)[-1500:]


def main():
    if DST.exists():
        held = [p for d in ('Data', '今天') for p in (DST/d).rglob('*') if p.is_file()]   # QA/ only holds the package tests' own output
        assert not held, f'{DST} already holds field data ({len(held)} files); refusing to touch it'
        assert '--replace' in sys.argv, f'{DST} exists; pass --replace to rebuild it'
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns('__pycache__', 'Data', 'QA', '今天', 'package_manifest.json'))
    for d in DATA_DIRS: (DST/'Data'/d).mkdir(parents=True, exist_ok=True)
    (DST/'QA').mkdir(exist_ok=True); (DST/'今天').mkdir(exist_ok=True)
    shutil.copyfile(MODEL, DST/'Model/venue_free_committee_1023.joblib')
    b = joblib.load(MODEL); digest = sha256(DST/'Model/venue_free_committee_1023.joblib'); n = b['training_events']
    S = json.loads(MODEL.with_suffix('.json').read_text(encoding='utf-8')); days = ','.join(S['days'])
    old = re.search(r"MODEL_SHA256 = '([0-9a-f]{64})'", (SRC/'Program/seventh_core.py').read_text(encoding='utf-8')).group(1)
    patch('Program/seventh_core.py', [(old, digest), ('TRAINING_EVENTS = 27132', f'TRAINING_EVENTS = {n}'), ("MODEL_NAME = '第4～10次", "MODEL_NAME = '10/09～11/10 ")])
    patch('Program/freeze_seventh_package.py', [(old, digest), ("bundle['training_events']==27132", f"bundle['training_events']=={n}"), ('training_events=27132', f'training_events={n}'),
                                                ("target_date='2026-11-10',prepared_date='2026-11-06'", f"target_date='{DATE}',prepared_date='{S['built']}'"),
                                                ("workflow_revision='2026-11-10-eleventh-R10-R4-R5-R6-R7-R8-R9-retrained-ninth-recipe'", f"workflow_revision='{DATE}-pkg{TAG}-{'-'.join(S['days'])}-retrained-ninth-recipe'"),
                                                ("training_scope='R10,R4,R5,R6,R7,R8,R9 songs", f"training_scope='{days} songs")])
    patch('Program/test_seventh_package.py', [("self.assertEqual(self.bundle['training_events'],27132)", f"self.assertEqual(self.bundle['training_events'],{n})"),
                                              ("out/replay_eleventh'", f"out/replay_{TAG}'")])
    patch('Program/seventh_collect.py', [('HoloGrip｜11/10 的包｜', 'HoloGrip｜11/12 的包｜')])
    patch('Program/seventh_native_gui.py', [('HoloGrip｜11/10 的包｜', 'HoloGrip｜11/12 的包｜'), ('11/10 的包｜打擊判斷＋10/09～11/06 資料訓練的模型', '11/12 的包｜打擊判斷＋10/09～11/10 資料訓練的模型')])
    py = sys.executable; rep = dict(src=SRC.name, dst=DST.name, model=str(MODEL), model_sha256=digest, training_events=n, days=S['days'], excluded=S['excluded'], steps={})
    rep['steps']['freeze'] = run([py, 'Program/freeze_seventh_package.py'], DST, f'log_{TAG}_freeze.log')
    rep['steps']['references'] = run([py, str(HERE/'score.py'), '--code', DST.name, '--days', 'R6,R7', '--tag', f'refs_{TAG}', '--refs', f'replay_{TAG}'], HERE, f'log_{TAG}_refs.log')
    rep['steps']['package_tests'] = run([py, 'Program/test_seventh_package.py'], DST, f'log_{TAG}_tests.log')
    rep['steps']['gui_test'] = run([py, 'Program/test_seventh_native_gui.py'], DST, f'log_{TAG}_gui.log')
    rep['steps']['self_check'] = run([py, 'Program/seventh_check.py'], DST, f'log_{TAG}_check.log')
    rep['passed'] = all(v[0] == 0 for v in rep['steps'].values())
    (HERE/'out'/f'package_{TAG}.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    for k, (code, tail) in rep['steps'].items(): print(k, 'exit', code, '|', tail.strip().splitlines()[-1][:200] if tail.strip() else '', flush=True)
    print('PASSED' if rep['passed'] else 'FAILED', DST, digest[:12], n, flush=True)


if __name__ == '__main__':
    main()
