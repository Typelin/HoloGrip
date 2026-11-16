"""Cleanest one-shot drum sounds from every field video, labelled by that day's MIDI (no sound classifier).

For each video audio (audio/<video>.wav) every MIDI file in the same folder as the video is aligned to the
audio onsets (time scale 0.90-1.15 and offset, same matcher as the glove alignment); a MIDI counts when at least
80 % of its notes land within 50 ms of an onset (a scale at the edge of the search range is treated as a false match);
the same hit reached through two versions of one MIDI is kept once; within one video, overlapping matches keep only
the highest match rate (similar drill rhythms produce false matches). Each aligned note (all notes, kick included) is a candidate when
  - no other MIDI note within 50 ms (not a chord) and the detected onset is within 30 ms,
  - pre_db  <= -20 dB (cymbals -8 dB): level 120..10 ms before the hit relative to the hit (ring from earlier sounds),
  - snr_db  >= 25 dB (cymbals 15 dB): hit level relative to the recording's quiet floor.
  Cymbals almost always ring over earlier sounds in songs (median pre_db about -2 dB), hence the looser gates.
Per drum: rank by ring length (time to the next onset, capped at 2.0 s for cymbals, 1.0 s for drums), then snr;
cut the top 3 (at most 2 per video) from 5 ms before the onset; trimmed 60 dB below peak, 30 % cosine fade-out,
peak -1 dBFS. Inputs read-only. Output: samples_best/<drum>_<n>.wav, index.json, out/midi_video_match.json,
out/samples_best.png."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; DESK = ROOT.parent
sys.path[:0] = [str(HERE), str(ROOT/'Research/NextRound_1105'), str(ROOT/'Research/AutoLabel_1027')]
import soundfile as sf, mido
import audio_1112 as A, autolabel_1027 as AL
from cut_samples_1112 import NOTE, FILE, CYM
from best_samples import rms, cut, plot, TARGET
OUT = HERE/'samples_best'
VIDEO_DIR = {'20261106_110612': '11_6/video', '20261106_112458': '11_6/video', '20261105_124817': 'HoloGrip/第十次的包_暫定/今天',
             '20261105_130038': 'HoloGrip/第十次的包_暫定/今天', '20261027_115924': '今天', '20261027_123510': '今天',
             '1790679188919': 'HoloGrip/第三次的包', '1791876567319': 'HoloGrip/第五次的包_暫定2026-10-13/我放這',
             '1791876640519': 'HoloGrip/第五次的包_暫定2026-10-13/我放這', '1791876666091': 'HoloGrip/第五次的包_暫定2026-10-13/我放這',
             '1792133713125': 'HoloGrip/第六次的包_暫定2026-10-16/今天', '1792133741038': 'HoloGrip/第六次的包_暫定2026-10-16/今天',
             '1792133773038': 'HoloGrip/第六次的包_暫定2026-10-16/今天', '1792133954572': 'HoloGrip/第六次的包_暫定2026-10-16/今天',
             '1791430739123': 'HoloGrip/第四次的包/東西我放這', '20261112_113843': 'HoloGrip/11-12的包/今天/雲端_11-12'}


def midi_notes(path):
    t = 0.0; out = []
    try:
        for m in mido.MidiFile(str(path)):
            t += m.time
            if m.type == 'note_on' and m.velocity > 0 and m.note in NOTE: out.append((t*1000, m.note, m.velocity))
    except Exception:
        return []
    return out


def main():
    match = []; cand = []
    for vid, sub in VIDEO_DIR.items():
        wav = HERE/'audio'/f'{vid}.wav'
        y, sr = sf.read(wav, dtype='float32'); y = y/max(np.max(np.abs(y)), 1e-9); ons, _, _ = A.onsets(y, sr)
        fr = int(0.05*sr); floor = np.percentile([rms(y[i:i+fr]) for i in range(0, len(y)-fr, fr)], 10)
        mids = [p for p in (DESK/sub).glob('*.mid')] + [p for p in (DESK/sub).parent.glob('*.mid')] if sub.endswith('video') else list((DESK/sub).glob('*.mid'))
        found = []
        for mp in mids:
            N = midi_notes(mp)
            if len(N) < 20: continue
            mt = np.array([n[0] for n in N])
            s, off, rate = max(((s,)+tuple(AL.align(mt*s, ons)) for s in np.arange(0.90, 1.15, 0.0005)), key=lambda x: x[2])
            at = mt*s+off; inside = np.mean((at > 0) & (at < len(y)/sr*1000))
            if rate < 0.8 or inside < 0.95 or not 0.905 <= s <= 1.145: continue   # edge-of-range scale = false match
            found.append((rate, mp, N, at, s, off))
        # one stretch of the video belongs to one MIDI: an overlapping match with a lower rate is a false match
        # (similar drill rhythms) or another version of the same MIDI
        kept = []
        for rate, mp, N, at, s, off in sorted(found, key=lambda f: -f[0]):
            a0, a1 = at.min(), at.max()
            if any(min(a1, b1)-max(a0, b0) > 0.2*(a1-a0) for b0, b1 in [(k[3].min(), k[3].max()) for k in kept]): continue
            kept.append((rate, mp, N, at, s, off))
        for rate, mp, N, at, s, off in kept:
            match.append(dict(video=vid, midi=mp.name, notes=len(N), scale=round(float(s), 4), offset_ms=round(float(off)), rate=round(float(rate), 3)))
            print('match', vid, mp.name, len(N), round(float(s), 4), round(float(rate), 3), flush=True)
            for i, (tm, note, vel) in enumerate(N):
                t0 = at[i]; others = np.delete(at, i)
                if len(others) and np.min(np.abs(others-t0)) <= 50: continue
                d = A.nearest(ons, np.array([t0]))[0]
                if abs(d) > 30: continue
                t = t0+d; k = int(np.searchsorted(ons, t+1)); j = int(t*sr/1000)
                if j < int(0.15*sr) or j+int(0.1*sr) >= len(y): continue
                after = (ons[k]-t) if k < len(ons) else (len(y)/sr*1000-t)
                hit = rms(y[j:j+int(0.03*sr)]); pre = rms(y[j-int(0.12*sr):j-int(0.01*sr)])
                pre_db = 20*np.log10(pre/hit); snr = 20*np.log10(hit/floor)
                cym = NOTE[note] in CYM
                if pre_db > (-8 if cym else -20) or snr < (15 if cym else 25): continue
                cand.append(dict(video=vid, midi=mp.name, t_ms=float(t), drum=NOTE[note], note=note, velocity=vel, prob=1.0,
                                 after_ms=float(after), pre_db=float(pre_db), snr_db=float(snr)))
    uniq = {}
    for c in cand: uniq.setdefault((c['video'], round(c['t_ms']/20)), c)   # same hit reached through two MIDI versions
    cand = list(uniq.values())
    (HERE/'out/midi_video_match.json').write_text(json.dumps(dict(match=match, candidates=cand), ensure_ascii=False, indent=1), encoding='utf-8')
    print('candidates', dict(Counter(c['drum'] for c in cand)), flush=True)
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob('*.wav'): f.unlink()
    index = {}
    for d in FILE:
        pool = sorted([c for c in cand if c['drum'] == d], key=lambda c: (-min(c['after_ms'], TARGET[d]), -c['snr_db']))
        pick = []; per = Counter()
        for c in pool:
            if per[c['video']] >= 2: continue
            pick.append(c); per[c['video']] += 1
            if len(pick) == 3: break
        index[d] = []
        for n, c in enumerate(pick, 1):
            x, sr = cut(c); name = f'{FILE[d]}_{n}.wav'; sf.write(OUT/name, x, sr, subtype='PCM_16')
            index[d].append(dict(file=name, video=c['video'], midi=c['midi'], midi_note=c['note'], velocity=c['velocity'],
                                 video_time_s=round(c['t_ms']/1000, 3), length_ms=round(len(x)/sr*1000),
                                 ring_ms=round(min(c['after_ms'], TARGET[d])), pre_db=round(c['pre_db'], 1), snr_db=round(c['snr_db'], 1)))
        print(d, len(pool), [(e['file'], e['video'], e['length_ms'], e['velocity']) for e in index[d]], flush=True)
    (OUT/'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')
    plot(index)


if __name__ == '__main__':
    main()
