"""Derived MIDI copies for 11/10 afternoon (人員3) -> out/midi_R12/. Originals (今天/雲端_11-10-2) are only read.
Ableton exports carry no tempo meta, so readers assume 120 BPM. Two of them were recorded in a 110 BPM project
(measured scale 1.0905-1.0910 against the glove clock); those copies get a 110 BPM set_tempo at tick 0.
Single-hand drills are renamed so the name ends with the hand (autolabel_day.hand_of)."""
import shutil
from pathlib import Path
import mido
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]/'11-10的包/今天/雲端_11-10-2'; DST = HERE/'out/midi_R12'; DST.mkdir(parents=True, exist_ok=True)
P = 'STEM 計畫-電子鼓數據搜集(1110)-'
JOBS = [('1110 LHM fixed.mid', '高音轉中音 L.mid', None), ('1110 LMR fixed.mid', '中音轉Ride L.mid', None), ('1110 LRG fixed.mid', 'Ride轉落地 L.mid', None),
        (P+'歌曲1 fixed5.0 .als.mid', '歌曲1.mid', 110), (P+'歌曲2 fixed.als.mid', '四鼓.mid', None),
        (P+'測試1 fixed.als.mid', '前測_1452.mid', 110), (P+'測試2.als.mid', '前測_1508.mid', None), ('1101 測試3 fixed.mid', '前測_1537_通鼓.mid', None)]
for src, dst, bpm in JOBS:
    if bpm is None: shutil.copyfile(SRC/src, DST/dst)
    else:
        m = mido.MidiFile(SRC/src); assert not any(x.type == 'set_tempo' for t in m.tracks for x in t)
        m.tracks[0].insert(0, mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0)); m.save(DST/dst)
    print(dst, '<-', src, bpm or '')
