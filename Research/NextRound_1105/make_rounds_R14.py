"""out/rounds_R14.json = rounds_R13.json (R4..R13, used for model g6) + R14 (11/17, 人員3).
R14 training = song 1 + the two single-hand drills (高音 Tom / Ride / 落地 Tom, left and right); the pretest is exam only.
Pairing checked with score_live_1117.py (out/log_live_1117.log): every pair aligns at time scale 1.0000."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
c = json.loads((H/'out/rounds_R13.json').read_text(encoding='utf-8'))
M = 'Data/模型實測/'; R = 'S20261117_P01_'
D3 = ['高音 Tom', 'Ride', '落地 Tom']
c['R14'] = {
 '_who': '人員3, 11/17 (files say P01). MIDI = derived copies (make_midi_R14.py), time scale 1.0.',
 'enabled': True, 'date': '20261117', 'package': '11-17的包', 'midi_dir': (H/'out/midi_R14').as_posix(), 'label_dir': 'out/labels_R14',
 'pretest_midi': ['前測.mid'],
 'pretest_raw': [M+'S20261117_P01_sevendrum_take02_20261117_095244_190038/S20261117_P01_sevendrum_take02_raw_100hz_20261117_095244.csv'],
 'single': {'高Ride落地 L': [D3, R+'L_H_R_G_take01_raw_100hz_20261117_111012'],
            '高Ride落地 R': [D3, R+'R_H_R_G_take01_raw_100hz_20261117_111810']},
 'songs': {'歌曲1': ['song', R+'song01_take01_raw_100hz_20261117_104001']},
 'void': ['094903 pretest take01 and 101856 take03: no matching MIDI (1117-1 LR 20 aligns only to 095244)',
          '110910: 37 s false start of the left drill', '11:27-11:49 sevendrum sessions: no MIDI']}
(H/'out/rounds_R14.json').write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding='utf-8'); print(list(c))
