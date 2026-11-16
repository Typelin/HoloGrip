"""Cut one-shot drum samples (wav) out of the 11/12 video audio, for later playback of judged hits.

Uses the song 1 MIDI (all notes, including the kick, MIDI 36) aligned to the audio by audio_1112.py.
For every drum it keeps the hits with the longest silence after them (no other MIDI note from 30 ms before the hit
until the cut ends), picks the 3 strongest of the most isolated ones, and cuts from 5 ms before the detected onset.
Length: up to 0.6 s for drums, 1.2 s for cymbals (shorter when the next note comes earlier); 40 % cosine fade-out;
each file peak-normalised to -1 dBFS (loudness by hit strength is left to the player).
Input is read-only. Output: samples_1112/<drum>_<n>.wav (48 kHz mono 16-bit) and samples_1112/index.json."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'Research/NextRound_1105'), str(ROOT/'Research/AutoLabel_1027')]
import soundfile as sf, mido
import audio_1112 as A
NOTE = {36: '大鼓', 38: '小鼓', 40: '小鼓', 37: '小鼓', 42: 'Hi-Hat', 44: 'Hi-Hat', 46: 'Hi-Hat', 49: 'Crash', 57: 'Crash',
        48: '高音 Tom', 50: '高音 Tom', 45: '中音 Tom', 47: '中音 Tom', 51: 'Ride', 59: 'Ride', 53: 'Ride', 43: '落地 Tom', 41: '落地 Tom'}
FILE = {'大鼓': 'kick', '小鼓': 'snare', 'Hi-Hat': 'hihat', 'Crash': 'crash', '高音 Tom': 'tom_high', '中音 Tom': 'tom_mid',
        'Ride': 'ride', '落地 Tom': 'tom_floor'}
CYM = {'Hi-Hat', 'Crash', 'Ride'}
OUT = HERE/'samples_1112'


def main():
    y, sr = sf.read(A.WAV, dtype='float32')
    ons, _, _ = A.onsets(y/np.max(np.abs(y)), sr)
    r = json.loads((HERE/'out/audio_1112.json').read_text(encoding='utf-8'))
    t = 0.0; N = []
    for m in mido.MidiFile(str(A.MID)):
        t += m.time
        if m.type == 'note_on' and m.velocity > 0: N.append((t*1000*r['audio_scale']+r['audio_offset_ms'], m.note, m.velocity))
    N.sort(); T = np.array([n[0] for n in N])
    OUT.mkdir(exist_ok=True); index = {}
    for drum in FILE:
        maxlen = 1200 if drum in CYM else 600; cand = []
        for i, (tm, note, vel) in enumerate(N):
            if NOTE.get(note) != drum or tm < 100 or tm > len(y)/sr*1000-maxlen: continue
            before = tm-T[i-1] if i else 1e9
            nxt = T[i+1]-tm if i+1 < len(T) else 1e9
            if before < 30: continue
            d = A.nearest(ons, np.array([tm]))[0]
            if abs(d) > 30: continue
            cand.append((min(nxt, maxlen), vel, tm+d, note))
        if not cand: index[drum] = []; continue
        best = max(c[0] for c in cand); pool = [c for c in cand if c[0] >= min(best, 0.8*maxlen)] or cand
        pick = sorted(pool, key=lambda c: (-c[1], -c[0]))[:3]; index[drum] = []
        for n, (L, vel, t0, note) in enumerate(pick, 1):
            a = int((t0-5)/1000*sr); b = a+int((L-10)/1000*sr); x = y[a:b].copy()
            f = int(0.4*len(x)); x[-f:] *= 0.5*(1+np.cos(np.linspace(0, np.pi, f)))
            x *= 10**(-1/20)/max(np.max(np.abs(x)), 1e-9)
            name = f'{FILE[drum]}_{n}.wav'; sf.write(OUT/name, x, sr, subtype='PCM_16')
            index[drum].append(dict(file=name, midi_note=note, velocity=vel, length_ms=round(len(x)/sr*1000), video_time_s=round(t0/1000, 3)))
    (OUT/'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')
    for d, v in index.items(): print(d, [(e['file'], e['length_ms'], e['velocity']) for e in v])


if __name__ == '__main__':
    main()
