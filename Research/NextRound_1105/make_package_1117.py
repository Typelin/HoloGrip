"""Create `11-17的包` from `11-12的包` (same program; the collector's session name already follows today's date)
with model g6 retrained on 10/09-11/12 (R4..R13; R13 = 11/12 人員2 controlled drills + song 1, weight 2, 5 seeds).
Also the live-test window's default session name follows today's date (was fixed at S9_P01_sevendrum_take01).
Field data is never copied: Data/, QA/ and 今天/ start empty. Refuses to touch a destination that holds field data.

  HOLOGRIP_ROUNDS=out/rounds_R13.json python make_package_1117.py --model=out/model_g6.joblib [--replace]
"""
from __future__ import annotations
import json, re, shutil, sys
from pathlib import Path
import joblib
import make_package_1112 as P   # helpers: sha256, run, DATA_DIRS, ENV
HERE = P.HERE; ROOT = P.ROOT
SRC = ROOT/'11-12的包'; DST = ROOT/'11-17的包'; TAG = '1117'; DATE = '2026-11-17'
MODEL = HERE/next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--model=')), 'out/model_g6.joblib')


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
    frz = (SRC/'Program/freeze_seventh_package.py').read_text(encoding='utf-8')
    td = re.search(r"target_date='[^']*',prepared_date='[^']*'", frz).group(0); wr = re.search(r"workflow_revision='[^']*'", frz).group(0)
    ts = re.search(r"training_scope='[^ ]*", frz).group(0)
    patch('Program/seventh_core.py', [(old, digest), (f'TRAINING_EVENTS = {n0}', f'TRAINING_EVENTS = {n}'), ("MODEL_NAME = '10/09～11/10 ", "MODEL_NAME = '10/09～11/12 ")])
    patch('Program/freeze_seventh_package.py', [(old, digest), (f"bundle['training_events']=={n0}", f"bundle['training_events']=={n}"), (f'training_events={n0}', f'training_events={n}'),
                                                (td, f"target_date='{DATE}',prepared_date='{S['built']}'"),
                                                (wr, f"workflow_revision='{DATE}-pkg{TAG}-{'-'.join(S['days'])}-retrained-ninth-recipe'"),
                                                (ts, f"training_scope='{days}")])
    patch('Program/test_seventh_package.py', [(f"self.assertEqual(self.bundle['training_events'],{n0})", f"self.assertEqual(self.bundle['training_events'],{n})"),
                                              ("out/replay_1112'", f"out/replay_{TAG}'")])
    patch('Program/seventh_collect.py', [('HoloGrip｜11/12 的包｜', 'HoloGrip｜11/17 的包｜')])
    patch('Program/seventh_native_gui.py', [('HoloGrip｜11/12 的包｜', 'HoloGrip｜11/17 的包｜'),
                                            ('11/12 的包｜打擊判斷＋10/09～11/10 資料訓練的模型', '11/17 的包｜打擊判斷＋10/09～11/12 資料訓練的模型'),
                                            ("self.session=tk.StringVar(value='S9_P01_sevendrum_take01')",
                                             "self.session=tk.StringVar(value=f\"S{__import__('time').strftime('%Y%m%d')}_P01_sevendrum_take01\")")])
    k = len(b['portable'])   # g6: 5 seeds per half -> 10 committee members (the runtime averages any number)
    patch('Program/seventh_core.py', [("len(bundle['portable']) != 6", f"len(bundle['portable']) != {k}")])
    patch('Program/freeze_seventh_package.py', [("assert len(bundle['portable'])==6", f"assert len(bundle['portable'])=={k}")])
    patch('Program/test_seventh_package.py', [("self.assertEqual(len(self.bundle['portable']),6)", f"self.assertEqual(len(self.bundle['portable']),{k})")])
    # 'start' re-verified the package by reloading the whole model (about 6 s with 6 members, 11 s with 10).
    # Split the file/fingerprint checks into verify_files(): 'start' re-checks every fingerprint (about 0.05 s)
    # and keeps the bundle that was loaded and validated when the service opened.
    patch('Program/seventh_core.py', [
        ('''def load_components():
    manifest = json.loads''',
         '''def verify_files():
    """Fingerprint checks of the frozen model, detector config and every runtime file (no model load)."""
    manifest = json.loads'''),
        ('''        raise ValueError('不是本次凍結的場地無關模型')
    bundle = joblib.load(MODEL)''',
         '''        raise ValueError('不是本次凍結的場地無關模型')
    return manifest, config


def load_components():
    manifest, config = verify_files()
    bundle = joblib.load(MODEL)''')])
    patch('Program/seventh_live.py', [("self.ready();core.load_components()", "self.ready();core.verify_files()")])
    py = sys.executable; rep = dict(src=SRC.name, dst=DST.name, model=str(MODEL), model_sha256=digest, training_events=n, days=S['days'], excluded=S['excluded'], steps={})
    rep['steps']['freeze'] = P.run([py, 'Program/freeze_seventh_package.py'], DST, f'log_{TAG}_freeze.log')
    rep['steps']['references'] = P.run([py, str(HERE/'score.py'), '--code', DST.name, '--days', 'R6,R7', '--tag', f'refs_{TAG}', '--refs', f'replay_{TAG}'], HERE, f'log_{TAG}_refs.log')
    rep['steps']['package_tests'] = P.run([py, 'Program/test_seventh_package.py'], DST, f'log_{TAG}_tests.log')
    rep['steps']['gui_test'] = P.run([py, 'Program/test_seventh_native_gui.py'], DST, f'log_{TAG}_gui.log')
    rep['steps']['self_check'] = P.run([py, 'Program/seventh_check.py'], DST, f'log_{TAG}_check.log')
    rep['passed'] = all(v[0] == 0 for v in rep['steps'].values())
    (HERE/'out'/f'package_{TAG}.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    for k, (code, tail) in rep['steps'].items(): print(k, 'exit', code, '|', tail.strip().splitlines()[-1][:200] if tail.strip() else '', flush=True)
    print('PASSED' if rep['passed'] else 'FAILED', DST, digest[:12], n, flush=True)


if __name__ == '__main__':
    main()
