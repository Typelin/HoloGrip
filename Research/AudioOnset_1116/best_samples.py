"""Pick the cleanest one-shot drum sounds from every field video (playback samples).

1. Train a sound classifier (8 classes: 7 drums + kick) on the 11/12 video, labelled by song 1 MIDI
   (single notes only: no other note within 50 ms), using the features of audio_1112.py. XGBoost on the GPU.
2. Scan every extracted video audio (audio/*.wav): for each onset measure
     after  = time to the next onset (how long the sound can ring)
     pre_db = level 120..10 ms before the onset relative to the hit (bleed from earlier sounds)
     snr_db = hit level relative to the recording's quiet floor (10th percentile of 50 ms RMS)
   and classify it. Keep hits with class probability >= 0.9, pre_db <= -30 dB and snr_db >= 30 dB.
3. Per drum rank by ring length (target 2.0 s for cymbals, 1.0 s for drums), then snr; cut the top 3
   from 5 ms before the onset until the next onset or the target length, trimmed where the sound falls 60 dB
   below its peak, 30 % cosine fade-out, peak -1 dBFS.
Inputs read-only. Output: samples_best/<drum>_<n>.wav, samples_best/index.json, out/samples_best.png,
out/best_candidates.json."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'Research/NextRound_1105'), str(ROOT/'Research/AutoLabel_1027')]
import soundfile as sf, mido
import audio_1112 as A
from cut_samples_1112 import NOTE, FILE, CYM
DRUMS = list(FILE)                       # 大鼓 first, then the 7 drums
OUT = HERE/'samples_best'
TARGET = {d: (2000 if d in CYM else 1000) for d in DRUMS}
GATE = dict(prob=0.9, pre_db=-30.0, snr_db=30.0)


def rms(x):
    return float(np.sqrt(np.mean(x**2))+1e-12)


def train():
    import xgboost as xgb
    y, sr = sf.read(A.WAV, dtype='float32'); y = y/np.max(np.abs(y)); ons, _, _ = A.onsets(y, sr)
    r = json.loads((HERE/'out/audio_1112.json').read_text(encoding='utf-8')); t = 0.0; N = []
    for m in mido.MidiFile(str(A.MID)):
        t += m.time
        if m.type == 'note_on' and m.velocity > 0 and m.note in NOTE: N.append((t*1000*r['audio_scale']+r['audio_offset_ms'], NOTE[m.note]))
    N.sort(); T = np.array([n[0] for n in N]); X = []; Y = []
    for i, (tm, d) in enumerate(N):
        gap = min(tm-T[i-1] if i else 1e9, T[i+1]-tm if i+1 < len(T) else 1e9)
        if gap <= 50 or not 100 < tm < len(y)/sr*1000-150: continue
        dd = A.nearest(ons, np.array([tm]))[0]
        if abs(dd) > 30: continue
        f = A.features(y, sr, tm+dd)
        if f is not None: X.append(f); Y.append(DRUMS.index(d))
    X = np.array(X); Y = np.array(Y); w = np.array([1.0/np.sum(Y == c) for c in Y]); w *= len(w)/w.sum()
    m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                          tree_method='hist', device='cuda', objective='multi:softprob', num_class=len(DRUMS))
    m.fit(X, Y, sample_weight=w)
    print('train', dict(Counter(DRUMS[c] for c in Y)), flush=True)
    return m


def scan(m, wav):
    y, sr = sf.read(wav, dtype='float32'); y = y/max(np.max(np.abs(y)), 1e-9); ons, _, _ = A.onsets(y, sr)
    fr = int(0.05*sr); floor = np.percentile([rms(y[i:i+fr]) for i in range(0, len(y)-fr, fr)], 10)
    out = []
    for k, t in enumerate(ons):
        i = int(t*sr/1000)
        if i < int(0.15*sr) or i+int(0.1*sr) >= len(y): continue
        after = (ons[k+1]-t) if k+1 < len(ons) else (len(y)/sr*1000-t)
        hit = rms(y[i:i+int(0.03*sr)]); pre = rms(y[i-int(0.12*sr):i-int(0.01*sr)])
        pre_db = 20*np.log10(pre/hit); snr = 20*np.log10(hit/floor)
        if pre_db > GATE['pre_db'] or snr < GATE['snr_db']: continue
        f = A.features(y, sr, t)
        if f is None: continue
        p = m.predict_proba(f[None, :])[0]; c = int(p.argmax())
        if p[c] < GATE['prob']: continue
        out.append(dict(video=wav.stem, t_ms=float(t), drum=DRUMS[c], prob=float(p[c]), after_ms=float(after), pre_db=float(pre_db), snr_db=float(snr)))
    return out


def cut(c):
    y, sr = sf.read(HERE/'audio'/f"{c['video']}.wav", dtype='float32')
    L = min(c['after_ms']-20, TARGET[c['drum']]); a = int((c['t_ms']-5)/1000*sr); x = y[a:a+int(L/1000*sr)].copy()
    env = np.array([rms(x[j:j+480]) for j in range(0, len(x)-480, 480)]); pk = env.max()
    below = np.where(env < pk*10**(-60/20))[0]
    if len(below) and below[0] > 10: x = x[:below[0]*480]
    f = int(0.3*len(x)); x[-f:] *= 0.5*(1+np.cos(np.linspace(0, np.pi, f)))
    return x*10**(-1/20)/max(np.max(np.abs(x)), 1e-9), sr


def main():
    m = train(); C = []
    for wav in sorted((HERE/'audio').glob('*.wav')):
        c = scan(m, wav); C += c
        print(wav.stem, len(c), dict(Counter(x['drum'] for x in c)), flush=True)
    (HERE/'out/best_candidates.json').write_text(json.dumps(C, ensure_ascii=False, indent=1), encoding='utf-8')
    OUT.mkdir(exist_ok=True); index = {}
    for d in DRUMS:
        pool = sorted([c for c in C if c['drum'] == d], key=lambda c: (-min(c['after_ms'], TARGET[d]), -c['snr_db']))
        index[d] = []
        for n, c in enumerate(pool[:3], 1):
            x, sr = cut(c); name = f'{FILE[d]}_{n}.wav'; sf.write(OUT/name, x, sr, subtype='PCM_16')
            index[d].append(dict(file=name, video=c['video'], video_time_s=round(c['t_ms']/1000, 3), length_ms=round(len(x)/sr*1000),
                                 prob=round(c['prob'], 3), pre_db=round(c['pre_db'], 1), snr_db=round(c['snr_db'], 1)))
        print(d, len(pool), [(e['file'], e['video'], e['length_ms'], e['prob']) for e in index[d]], flush=True)
    (OUT/'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')
    plot(index)


def plot(index):
    import librosa, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    files = [(d, e) for d, v in index.items() for e in v]
    fig, ax = plt.subplots(len(files), 2, figsize=(12, 1.5*len(files)))
    for i, (d, e) in enumerate(files):
        x, sr = sf.read(OUT/e['file']); t = np.arange(len(x))/sr*1000
        ax[i, 0].plot(t, x, lw=0.4, color='k'); ax[i, 0].set_xlim(0, 2000)
        ax[i, 0].set_ylabel(f"{e['file'][:-4]}\n{e['video']}", rotation=0, ha='right', fontsize=7)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=1024, hop_length=128)), ref=np.max)
        ax[i, 1].imshow(S, origin='lower', aspect='auto', extent=[0, len(x)/sr*1000, 0, sr/2000], cmap='magma', vmin=-60)
        ax[i, 1].set_xlim(0, 2000); ax[i, 1].set_ylim(0, 16)
    ax[-1, 0].set_xlabel('ms'); ax[-1, 1].set_xlabel('ms (0-16 kHz)'); fig.tight_layout(); fig.savefig(HERE/'out/samples_best.png', dpi=80)


if __name__ == '__main__':
    main()
