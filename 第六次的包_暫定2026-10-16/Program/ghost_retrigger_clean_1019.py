"""Uniform ghost-retrigger rule for MIDI answers (all recordings, MIDI-only).

A MIDI answer is a ghost retrigger when BOTH hold:
  - MIDI velocity < 20 (real strikes on this kit are ~50-95; retriggers are 3-4), and
  - the same MIDI note sounded 110-260 ms earlier (cymbal-swing retrigger at ~190 ms), or
    the same note is duplicated within 20 ms (simultaneous double trigger).
The rule reads only the MIDI file. It never looks at Raw, the detector or any model
output, and it is applied identically to every recording, including pretests.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

PROG = Path(__file__).resolve().parent
sys.path.insert(0, str(PROG))
import midi_label_pipeline as pipe
import research_pretest_zone_models_1019 as first

H = Path(r'C:/Users/Typelin_Station/Desktop/HoloGrip')
T6 = H/'第六次的包_暫定2026-10-16/今天'
MIDI = {
    'fourth_song03': H/'第四次的包/東西我放這/測試4 更新.mid',
    'fifth_song01': H/'第五次的包_暫定2026-10-13/我放這/測試2.mid',
    'fifth_song02': H/'第五次的包_暫定2026-10-13/我放這/FIXED/測試3.mid',
    'fifth_song03': H/'第五次的包_暫定2026-10-13/我放這/FIXED/測試4.mid',
    'sixth_S02': T6/'FIXED/10_16歌曲2.mid', 'sixth_S08': T6/'TomTom.mid',
    'sixth_S04': T6/'左手Hihat.mid', 'sixth_S05': T6/'左手Crash.mid',
    'sixth_S06': T6/'左手Ride.mid', 'sixth_S07': T6/'右手Crash.mid',
}
VEL_MAX, GAP_LO, GAP_HI, DUP_MS = 20, 110.0, 260.0, 20.0


def ghost_ids(truth, midi_path):
    ev = pipe.parse_midi(Path(midi_path), None)['events']
    by_zone = {}
    for e in ev: by_zone.setdefault(e.zone_name, []).append(e)
    ghosts, unmatched = set(), 0
    for g in truth:
        g = dict(g, midi_ms=float(g['midi_ms']))
        zone = by_zone.get(g['drum'], [])
        e = min(zone, key=lambda e: abs(e.time_ms-g['midi_ms']), default=None)
        if e is None or abs(e.time_ms-g['midi_ms']) > 1.0:
            unmatched += 1; continue
        cands = [q for q in zone if q.note == e.note]
        others = [q for q in cands if q is not e]
        dup = any(abs(q.time_ms-e.time_ms) <= DUP_MS for q in others)
        prev = [q for q in others if q.time_ms < e.time_ms-DUP_MS]
        gap = e.time_ms-max(q.time_ms for q in prev) if prev else None
        if e.velocity < VEL_MAX and (dup or (gap is not None and GAP_LO <= gap <= GAP_HI)):
            ghosts.add(g['id'])
    return ghosts, unmatched


def main():
    sessions = joblib.load(first.OLD_CACHE)
    base = joblib.load(first.OUT/'特徵快取.joblib')
    out = {}
    for s in sessions:
        g, u = ghost_ids(s['truth'], MIDI[s['name']])
        c = Counter(t['drum'] for t in s['truth'] if t['id'] in g)
        out[s['name']] = dict(answers=len(s['truth']), ghosts=sorted(g), n_ghost=len(g), by_drum=dict(c), unmatched_lookup=u)
        print(s['name'], len(s['truth']), 'ghost', len(g), dict(c), 'unmatched', u, flush=True)
    for t in base['tests']:
        g, u = ghost_ids(t['session']['truth'], t['session']['midi'])
        out['pretest:'+t['session']['name']] = dict(answers=len(t['session']['truth']), ghosts=sorted(g), n_ghost=len(g), unmatched_lookup=u)
        print('pretest', t['session']['name'], len(t['session']['truth']), 'ghost', len(g), 'unmatched', u, flush=True)
    out['rule'] = dict(velocity_below=VEL_MAX, same_note_gap_ms=[GAP_LO, GAP_HI], same_note_duplicate_ms=DUP_MS, midi_only=True)
    dst = PROG.parent/'Data/場地無關模型_1019/幽靈鼓名單_規則固定.json'
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
