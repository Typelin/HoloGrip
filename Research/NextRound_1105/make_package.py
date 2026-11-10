"""Create the next field package from the 9th package with a newly built model, freeze it and run its tests.

Program code is copied unchanged except the frozen-model identity (fingerprint, training count,
names shown in the windows, manifest dates) and the test reference folder. Field data is never
copied: Data/ starts empty, and an existing destination with any file under Data/ is refused.

  python make_package.py --model out/model_tenth.joblib --name 第十次的包_暫定 --ordinal 第十次 --date 2026-11-06
Steps: copy -> install model -> patch identity -> freeze manifest -> replay references -> package tests
-> GUI test. Writes out/package_<name>.json with every step's result.
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, subprocess, sys, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--name', required=True, help='destination folder under HoloGrip/')
ap.add_argument('--ordinal', required=True, help='e.g. 第十次 (window titles)')
ap.add_argument('--date', required=True, help='field date YYYY-MM-DD')
ap.add_argument('--src', default='第九次的包_暫定')
ap.add_argument('--tag', default='', help='short name for logs / references (default: folder name before 的包)')
ap.add_argument('--replace', action='store_true', help='rebuild an existing destination (only when its Data/ has no files)')
ap.add_argument('--snare-relaxed-top', type=int, default=0,
                help='automatic anchor: when neither existing rule holds and the first N hits are tight, anchor on the snare '
                     'if it is within the committee top K for every one of them (0 = off, the 9th package rule)')
ARGS = ap.parse_args()

import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT/ARGS.src; DST = ROOT/ARGS.name
MODEL = Path(ARGS.model) if Path(ARGS.model).is_absolute() else HERE/ARGS.model
DATA_DIRS = ['MIDI', 'Raw', '全程監聽', '對齊與異常報告', '模型實測']
ENV = dict(os.environ, PYTHONIOENCODING='utf-8')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def patch(rel, pairs):
    """Exact replacements; each old text must occur exactly once (fails loudly when the source changed)."""
    p = DST/rel; s = p.read_bytes().decode('utf-8')
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, f'{rel}: expected 1 occurrence of {old!r}, found {n}'
        s = s.replace(old, new)
    p.write_bytes(s.encode('utf-8'))


def relax_anchor(k):
    """Third anchor rule (after 'snare top-1' and 'one consistent drum'): snare within the top k for all first N hits."""
    rel = 'Program/venue_free_runtime_1023.py'; s = (DST/rel).read_bytes().decode('utf-8'); nl = '\r\n' if '\r\n' in s else '\n'
    l1 = "                if np.all(top == top[0]) and P0[:self.anchor, top[0]].mean() >= ANCHOR_RULE['any_drum_min_score']: drum = int(top[0])"
    l2 = "            if np.all(top == top[0]) and P0[:, top[0]].mean() >= ANCHOR_RULE['any_drum_min_score']: drum = int(top[0])"
    patch(rel, [("ANCHOR_RULE = dict(snare_in_top=1, max_pose_deg=8.0, any_drum_min_score=0.7)",
                 f"ANCHOR_RULE = dict(snare_in_top=1, max_pose_deg=8.0, any_drum_min_score=0.7, snare_relaxed_top={k})"),
                (l1, l1+nl+"                if drum is None and all(self.anchor_drum in np.argsort(-P0[j])[:ANCHOR_RULE['snare_relaxed_top']] for j in range(self.anchor)): drum = self.anchor_drum"),
                (l2, l2+nl+"            if drum is None and all(self.anchor_drum in np.argsort(-p0)[:ANCHOR_RULE['snare_relaxed_top']] for _, p0, _, _ in pend): drum = self.anchor_drum"),
                ("(e.g. a take that opens on the Hi-Hat). Otherwise they enter normally",
                 f"(e.g. a take that opens on the Hi-Hat). If neither holds but the N poses are still tight, they anchor on the{nl}"
                 f"snare when it is within the committee's top {k} for every one of them. Otherwise they enter normally")])


def rounds_label(days):
    nums = sorted(int(d[1:]) for d in days)
    return f'第{nums[0]}～{nums[-1]}次' if nums == list(range(nums[0], nums[-1]+1)) else '第' + '、'.join(map(str, nums)) + '次'


def run(cmd, cwd, log):
    r = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, encoding='utf-8', errors='replace')
    (HERE/'out'/log).write_text(r.stdout+'\n--- stderr ---\n'+r.stderr, encoding='utf-8')
    return r.returncode, (r.stdout+r.stderr)[-1500:]


def main():
    report = dict(src=ARGS.src, dst=ARGS.name, model=str(MODEL), steps={})
    if DST.exists():
        held = [p for p in (DST/'Data').rglob('*') if p.is_file()]
        assert not held, f'{DST} already holds field data ({len(held)} files); refusing to touch it'
        assert ARGS.replace, f'{DST} exists; pass --replace to rebuild it'
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns('__pycache__', 'Data', 'QA'))
    for d in DATA_DIRS: (DST/'Data'/d).mkdir(parents=True, exist_ok=True)
    (DST/'QA').mkdir(exist_ok=True)
    shutil.copyfile(MODEL, DST/'Model/venue_free_committee_1023.joblib')
    b = joblib.load(MODEL); digest = sha256(DST/'Model/venue_free_committee_1023.joblib'); n = b['training_events']
    summary = json.loads(MODEL.with_suffix('.json').read_text(encoding='utf-8'))
    label = rounds_label(summary['days']); old_sha = re.search(r"MODEL_SHA256 = '([0-9a-f]{64})'", (SRC/'Program/seventh_core.py').read_text(encoding='utf-8')).group(1)
    tag = ARGS.tag or ARGS.name.split('的包')[0]
    patch('Program/seventh_core.py', [(old_sha, digest), ('TRAINING_EVENTS = 19358', f'TRAINING_EVENTS = {n}'),
                                      ("MODEL_NAME = '第4～8次", f"MODEL_NAME = '{label}")])
    patch('Program/freeze_seventh_package.py', [(old_sha, digest), ("bundle['training_events']==19358", f"bundle['training_events']=={n}"),
                                                ('training_events=19358', f'training_events={n}'),
                                                ("schema='hologrip-ninth-field-package-v1',target_date='undecided',prepared_date='2026-10-27'",
                                                 f"schema='hologrip-ninth-field-package-v1',target_date='{ARGS.date}',prepared_date='{summary.get('built', ARGS.date)}'"),
                                                ("workflow_revision='2026-10-27-ninth-rounds4to8-weighted-ensemble-auto-anchor'",
                                                 f"workflow_revision='{ARGS.date}-{tag}-{'-'.join(summary['days'])}-retrained-ninth-recipe'"),
                                                ("training_scope='rounds 4-8 songs", f"training_scope='{','.join(summary['days'])} songs")])
    patch('Program/test_seventh_package.py', [("self.assertEqual(self.bundle['training_events'],19358)", f"self.assertEqual(self.bundle['training_events'],{n})"),
                                              ("ROOT/'Research/AutoLabel_1027/out/replay_9th'", f"ROOT/'Research/NextRound_1105/out/replay_{tag}'")])
    patch('Program/seventh_collect.py', [('HoloGrip｜第九次｜', f'HoloGrip｜{ARGS.ordinal}｜')])
    patch('Program/seventh_check.py', [('Software PASS: 9th package - rounds 4-8 committee',
                                        f"Software PASS: {tag} package (9th recipe) - {','.join(summary['days'])} committee")])
    patch('Program/seventh_native_gui.py', [('HoloGrip｜第九次 ', f'HoloGrip｜{ARGS.ordinal} '),
                                            ('第九次｜打擊判斷＋第4～8次模型', f'{ARGS.ordinal}｜打擊判斷＋{label}模型')])
    if ARGS.snare_relaxed_top: relax_anchor(ARGS.snare_relaxed_top)
    readme = (SRC/'README_現場流程_ZH_TW.md').read_text(encoding='utf-8')
    field = readme[readme.index('## 現場操作'):]
    (DST/'README_現場流程_ZH_TW.md').write_text(
        f"# HoloGrip {ARGS.ordinal}現場包｜{ARGS.date}\n\n## 跟{ARGS.src.split('的包')[0]}包的差別\n"
        + (f"- 自動錨點多一條規則：前 5 下姿勢集中、但兩條原規則都不成立時，只要每一下小鼓都在模型前 {ARGS.snare_relaxed_top} 名，就以小鼓當錨點（前測開頭固定是小鼓）。其餘程式相同。\n"
           if ARGS.snare_relaxed_top else "- 程式與判斷邏輯完全相同（打擊判斷、雙模型加權、逐鼓位置校正、位置先驗、自動錨點、歸零方向保護、姿勢連續）。\n")
        + f"- 模型用 {label} 資料重新訓練（{n:,} 下；前測一律不進訓練），做法與第九次包相同（`Research/NextRound_1105/build_model.py`）。\n"
        f"- 模型指紋 `{digest[:12]}…`。\n\n## 成績\n見 `Research/NextRound_1105/out/score_*.md`（全部訓練與留一天分開列）。\n\n" + field, encoding='utf-8')
    py = sys.executable
    report['steps']['freeze'] = run([py, 'Program/freeze_seventh_package.py'], DST, f'log_{tag}_freeze.log')
    report['steps']['references'] = run([py, str(HERE/'score.py'), '--code', ARGS.name, '--days', 'R6,R7', '--tag', f'refs_{tag}',
                                         '--refs', f'replay_{tag}'], HERE, f'log_{tag}_refs.log')
    report['steps']['package_tests'] = run([py, 'Program/test_seventh_package.py'], DST, f'log_{tag}_tests.log')
    report['steps']['gui_test'] = run([py, 'Program/test_seventh_native_gui.py'], DST, f'log_{tag}_gui.log')
    report['model_sha256'] = digest; report['training_events'] = n; report['rounds'] = label
    report['passed'] = all(v[0] == 0 for v in report['steps'].values())
    (HERE/'out'/f'package_{tag}.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    for k, (code, tail) in report['steps'].items(): print(k, 'exit', code, '|', tail.strip().splitlines()[-1] if tail.strip() else '', flush=True)
    print('PASSED' if report['passed'] else 'FAILED', DST, flush=True)
    sys.exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
