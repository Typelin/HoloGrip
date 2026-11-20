"""Create `11-24的包` from `11-17的包` (same program and runtime: position-prior floor 0.1, fast 'start', dated session names)
with a model retrained on 10/09-11/17 (R4..R14; R14 = 11/17 人員3 song 1 + left single-hand drill; the right drill is left out,
see 現場紀錄_20261117_ZH_TW.md) and an opening gap on pose continuity (exp_gap_1124.py). Field data is never copied: Data/, QA/ and 今天/ start empty.
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
        if '\n' in old and s.count(old) == 0 and s.count(old.replace('\n', '\r\n')) == 1:   # CRLF source file
            old, new = old.replace('\n', '\r\n'), new.replace('\n', '\r\n')
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
    # Opening continuity gap (exp_gap_1124.py, 11/20): during the first 5 accepted hits of a hand, pose continuity keeps
    # the previous drum only if its score is within 0.5 of this hit's top score. On 11/17 one wrong first hit
    # (Hi-Hat 0.54 vs snare 0.34) was carried over hits the committee called snare at 0.85+.
    patch('Program/venue_free_runtime_1023.py', [
        ("CONTINUITY = dict(max_ms=600.0, max_deg=16.0, below_score=0.9, previous_in_top=2)",
         "CONTINUITY = dict(max_ms=600.0, max_deg=16.0, below_score=0.9, previous_in_top=2, opening_hits=5, opening_max_gap=0.5)\n\n\n"
         "def _opening_ok(q, previous_drum, accepted):\n"
         "    \"\"\"Opening gap: within the first opening_hits accepted hits, continuity needs the previous drum within opening_max_gap of the top score.\"\"\"\n"
         "    return accepted > CONTINUITY['opening_hits'] or q.max()-q[previous_drum] <= CONTINUITY['opening_max_gap']"),
        ("  3. pose continuity: if this hit's score is < 0.9, the same hand's previous accepted hit was within\n"
         "     600 ms with a zero-relative yaw/pitch within 16 deg, and that previous drum is among this hit's two\n"
         "     highest-scoring drums, it is the same drum as that previous hit.",
         "  3. pose continuity: if this hit's score is < 0.9, the same hand's previous accepted hit was within\n"
         "     600 ms with a zero-relative yaw/pitch within 16 deg, and that previous drum is among this hit's two\n"
         "     highest-scoring drums, it is the same drum as that previous hit. During the hand's first 5 accepted\n"
         "     hits the previous drum must also score within 0.5 of the top drum (one wrong opening hit is not carried)."),
        ("and prev[2] in np.argsort(-q)[:CONTINUITY['previous_in_top']] and float(np.hypot(*(Y[k]-prev[1]))) <= CONTINUITY['max_deg']):",
         "and prev[2] in np.argsort(-q)[:CONTINUITY['previous_in_top']] and _opening_ok(q, prev[2], k+1)\n"
         "                    and float(np.hypot(*(Y[k]-prev[1]))) <= CONTINUITY['max_deg']):"),
        ("                    and prev[2] in np.argsort(-q)[:CONTINUITY['previous_in_top']]\n                    and float(np.hypot(*(yp-prev[1])))",
         "                    and prev[2] in np.argsort(-q)[:CONTINUITY['previous_in_top']] and _opening_ok(q, prev[2], self.accepted[hand])\n"
         "                    and float(np.hypot(*(yp-prev[1])))")])
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
