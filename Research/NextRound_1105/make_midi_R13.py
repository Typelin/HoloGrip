"""Derived MIDI copies for 11/12 (人員2) -> out/midi_R13/. Originals (11-12的包/今天/雲端_11-12, from Drive 流音/11／12) are only read.
All 11/12 exports align at time scale 1.0000, so no tempo meta is added. Single-hand drills are renamed so the
name ends with the hand (autolabel_day.hand_of). Only the controlled drills (from L_H_M_R on) are used for training;
the earlier fast drills (L_M, R_M, L_M_R, R_M_R) are reference only and are not copied."""
import shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]/'11-12的包/今天/雲端_11-12'; DST = HERE/'out/midi_R13'; DST.mkdir(parents=True, exist_ok=True)
JOBS = [('11：12Tom1＼2Ride (左手).mid', '高中Ride L.mid'), ('11：12 Tom1：2 Ride (右手).mid', '高中Ride R.mid'),
        ('11：12Tom2 Ride Ftom(左手).mid', '中Ride落地 L.mid'), ('11：12Tom Ride Ftom(右手).mid', '中Ride落地 R.mid'),
        ('11：12歌曲1(修正.mid', '歌曲1.mid'), ('11：12.mid', '前測_1125.mid'), ('11：12 測試.mid', '前測_1130.mid')]
for src, dst in JOBS:
    shutil.copyfile(SRC/src, DST/dst); print(dst, '<-', src)
