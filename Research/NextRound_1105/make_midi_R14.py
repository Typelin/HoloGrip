"""Derived MIDI copies for 11/17 (人員3) -> out/midi_R14/. Originals (11-17的包/今天/雲端_11-17, from Drive 流音/11／17) are only read.
Single-hand drills are renamed so the name ends with the hand (autolabel_day.hand_of). Time scale is checked by score_live_1117.py."""
import shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]/'11-17的包/今天/雲端_11-17'; DST = HERE/'out/midi_R14'; DST.mkdir(parents=True, exist_ok=True)
JOBS = [('1117-3 L5 T1RidFT  (fixed).mid', '高Ride落地 L.mid'), ('1117-3 R5 T1RidFT  (fixed).mid', '高Ride落地 R.mid'),
        ('1117-2.1 All  (fixed).mid', '歌曲1.mid'), ('1117-1 LR 20.mid', '前測.mid')]
for src, dst in JOBS:
    shutil.copyfile(SRC/src, DST/dst); print(dst, '<-', src)
