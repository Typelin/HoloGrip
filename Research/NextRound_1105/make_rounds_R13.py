"""out/rounds_R13.json = rounds_1112.json (R4..R12, used for model e1) + R13 (11/12, 人員2).
R13 training = the four controlled drills + song 1; pretests are exam only; the fast drills are reference only."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
c = json.loads((H/'out/rounds_1112.json').read_text(encoding='utf-8'))
M = 'Data/模型實測/'; R = 'S20261112_'
c['R13'] = {
 '_who': '人員2, 11/12 (files say P01). MIDI = derived copies (make_midi_R13.py), time scale 1.0.',
 'enabled': True, 'date': '20261112', 'package': '11-12的包', 'midi_dir': (H/'out/midi_R13').as_posix(), 'label_dir': 'out/labels_R13',
 'pretest_midi': ['前測_1125.mid', '前測_1130.mid'],
 'pretest_raw': [M+'S9_P01_sevendrum_take01_20261112_112540_315869/S9_P01_sevendrum_take01_raw_100hz_20261112_112540.csv',
                 M+'S9_P01_sevendrum_take02_20261112_113018_315049/S9_P01_sevendrum_take02_raw_100hz_20261112_113018.csv'],
 'single': {'高中Ride L': [['高音 Tom', '中音 Tom', 'Ride'], R+'L_H_M_R_take01_raw_100hz_20261112_121424'],
            '高中Ride R': [['高音 Tom', '中音 Tom', 'Ride'], R+'R_H_M_R_take01_raw_100hz_20261112_121855'],
            '中Ride落地 L': [['中音 Tom', 'Ride', '落地 Tom'], R+'L_M_R_G_take01_raw_100hz_20261112_122405'],
            '中Ride落地 R': [['中音 Tom', 'Ride', '落地 Tom'], R+'R_M_R_G_take01_raw_100hz_20261112_122833']},
 'songs': {'歌曲1': ['song', R+'P01_song01_take01_raw_100hz_20261112_113853']},
 'void': ['L_M 115527, R_M 120107, L_M_R 120523, R_M_R 120924: fast continuous drills with body sway -> reference only',
          '113818: 5 s false start']}
(H/'out/rounds_R13.json').write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding='utf-8'); print(list(c))
