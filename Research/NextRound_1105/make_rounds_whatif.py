"""RESEARCH WHAT-IF ONLY (never shipped): would block-style data of 人員3 fix his left-hand toms?
out/rounds_whatif.json = rounds_1112 but the 15:37 tom pretest (block play, saved in song mode) is moved from the
exam list into R12's training takes. The other two R12 pretests (14:52, 15:08) stay exam only."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
c = json.loads((H/'out/rounds_1112.json').read_text(encoding='utf-8')); r = c['R12']
r['label_dir'] = 'out/labels_R12w'; r['pretest_midi'] = r['pretest_midi'][:2]; r['pretest_raw'] = r['pretest_raw'][:2]
r['songs']['前測_1537_通鼓'] = ['four_drum', 'S20261020_P02_song02_take01_raw_100hz_20261110_153748']
r['_who'] += ' WHAT-IF: 15:37 tom pretest used as training (research only).'
(H/'out/rounds_whatif.json').write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding='utf-8'); print('ok')
