"""out/rounds_1112.json = rounds_r10pre (R9, R10, R10b) + R11 (11/10 morning, 人員1) + R12 (11/10 afternoon, 人員3)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
c = json.loads((H/'out/rounds_r10pre.json').read_text(encoding='utf-8'))
c['R11'] = json.loads((H/'out/rounds_1110.json').read_text(encoding='utf-8'))['R11']; c['R11']['_who'] = '人員1, 11/10 morning'
M = 'Data/模型實測/'; R = 'S20261020_P02_'
c['R12'] = {
 '_who': '人員3, 11/10 afternoon (first time in training). MIDI = derived copies (make_midi_R12.py); two got the 110 BPM tempo meta.',
 'enabled': True, 'date': '20261110', 'package': '11-10的包', 'midi_dir': (H/'out/midi_R12').as_posix(), 'label_dir': 'out/labels_R12',
 'pretest_midi': ['前測_1452.mid', '前測_1508.mid', '前測_1537_通鼓.mid'],
 'pretest_raw': [M+'S9_P02_sevendrum_take01_20261110_145206_311269/S9_P02_sevendrum_take01_raw_100hz_20261110_145206.csv',
                 M+'S9_P02_sevendrum_take03_20261110_150827_064790/S9_P02_sevendrum_take03_raw_100hz_20261110_150827.csv',
                 'Data/Raw/'+R+'song02_take01_raw_100hz_20261110_153748.csv'],
 'single': {'高音轉中音 L': [['高音 Tom', '中音 Tom'], R+'L_H_M_take02_raw_100hz_20261110_155247'],
            '中音轉Ride L': [['中音 Tom', 'Ride'], R+'L_M_R_take01_raw_100hz_20261110_155804'],
            'Ride轉落地 L': [['Ride', '落地 Tom'], R+'L_R_G_take01_raw_100hz_20261110_160236']},
 'songs': {'歌曲1': ['song', R+'song01_take01_raw_100hz_20261110_135707'], '四鼓': ['four_drum', R+'song03_take01_raw_100hz_20261110_154120']},
 'void': ['13:36/13:39/13:42/13:44 pretests: no MIDI (pause segmentation only)', '12:44, 16:11, 16:13, 16:17 air-drum tests: no MIDI',
          '15:37 song02 = tom pretest saved in song mode -> exam only', 'L_H_M take01 155224, L_M_R take02 155746: aborted']}
(H/'out/rounds_1112.json').write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding='utf-8'); print(list(c))
