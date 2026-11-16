"""11/12 video audio vs song 1 MIDI: hit onsets, timing jitter, drum classification from multi-feature spectra.

Inputs (read-only): audio/20261112_113843.wav (mono 48 kHz, extracted from the phone video with ffmpeg),
song 1 MIDI (11-12的包/今天/雲端_11-12/11：12歌曲1(修正.mid), glove raw S20261112_P01_song01_take01 (113853).
  1. onsets: librosa spectral-flux onset envelope (hop 128 = 2.7 ms), peak picking, refined to the steepest
     energy rise within +-8 ms.
  2. audio<->MIDI alignment: global time scale + offset search (same matcher as the glove MIDI alignment).
  3. timing: residual of each MIDI hit vs the nearest audio onset, and vs the nearest glove peak (MIDI->CSV
     alignment), so both sensors are compared against the drum module's trigger times. A constant offset cannot
     be measured (no shared clock); only the spread (jitter) can.
  4. drum from sound: single notes only (no other note within 50 ms); features from the first 100 ms
     (log-mel in two windows, MFCC, centroid/bandwidth/rolloff/flatness/ZCR, energy decay, low-frequency peak);
     XGBoost (GPU) with 5 contiguous time-block folds (no neighbouring hits shared between train and test).
Writes out/audio_1112.txt, out/audio_1112.json and out/audio_1112_*.png."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; NR = ROOT/'Research/NextRound_1105'
sys.path[:0] = [str(NR), str(ROOT/'Research/AutoLabel_1027')]
import soundfile as sf, librosa
OUT = HERE/'out'; OUT.mkdir(exist_ok=True)
WAV = HERE/'audio/20261112_113843.wav'
MID = ROOT/'11-12的包/今天/雲端_11-12/11：12歌曲1(修正.mid'
RAWKEY = '113853'
ORDER = ['小鼓', 'Hi-Hat', 'Crash', '高音 Tom', '中音 Tom', 'Ride', '落地 Tom']
EN = {'小鼓': 'Snare', 'Hi-Hat': 'Hi-Hat', 'Crash': 'Crash', '高音 Tom': 'Tom1', '中音 Tom': 'Tom2', 'Ride': 'Ride', '落地 Tom': 'Floor'}


def onsets(y, sr):
    hop = 128
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_mels=128, fmax=16000, aggregate=np.median)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units='frames', backtrack=False,
                                    pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=0.05, wait=8)
    e = np.convolve(y**2, np.ones(48)/48, mode='same'); de = np.diff(np.log(e+1e-10))
    t = []
    for f in fr:
        c = f*hop; a, b = max(c-384, 1), min(c+384, len(de)-1)
        t.append((a+int(np.argmax(de[a:b])))/sr*1000.0)
    return np.array(sorted(t)), env, hop


def features(y, sr, t_ms):
    i = int(t_ms*sr/1000); seg = y[max(i-48, 0): i+int(0.1*sr)]
    if len(seg) < int(0.09*sr): return None
    S = np.abs(librosa.stft(seg, n_fft=1024, hop_length=128))**2
    mel = librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=sr, n_mels=48, fmax=16000))
    k = mel.shape[1]; a = mel[:, :max(2, k//3)].mean(1); b = mel[:, k//3:].mean(1)
    mf = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=sr, n_mels=64)), n_mfcc=20).mean(1)
    sp = [librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr).mean(), librosa.feature.spectral_bandwidth(S=np.sqrt(S), sr=sr).mean(),
          librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=sr).mean(), librosa.feature.spectral_flatness(S=np.sqrt(S)).mean(),
          librosa.feature.zero_crossing_rate(seg).mean()]
    r = librosa.feature.rms(y=seg, frame_length=512, hop_length=128)[0]; decay = np.log((r[-4:].mean()+1e-8)/(r[:4].max()+1e-8))
    spec = np.abs(np.fft.rfft(seg*np.hanning(len(seg)))); fq = np.fft.rfftfreq(len(seg), 1/sr); lo = (fq > 40) & (fq < 1000)
    return np.r_[a, b, b-a, mf, sp, decay, fq[lo][np.argmax(spec[lo])]].astype(np.float32)


def nearest(P, t):
    i = np.clip(np.searchsorted(P, t), 1, len(P)-1); j = np.where(np.abs(P[i]-t) < np.abs(P[i-1]-t), i, i-1)
    return P[j]-t


def main():
    import autolabel_1027 as AL, rounds_data as RD
    y, sr = sf.read(WAV, dtype='float32'); y = y/np.max(np.abs(y))
    ons, env, hop = onsets(y, sr)
    ev = [e for e in AL.legacy.midi(MID)['events'] if e['vel'] >= AL.GHOST_VEL and e['name'] in RD.K.DRUMS]
    mt = np.array([e['t'] for e in ev]); md = [e['name'] for e in ev]
    s, off, rate = max(((s,)+tuple(AL.align(mt*s, ons)) for s in np.arange(0.995, 1.005, 0.0001)), key=lambda x: x[2])
    at = mt*s+off                                               # MIDI notes on the audio timeline (ms)
    # events = MIDI notes merged within 30 ms (a chord sounds as one onset)
    grp = [];
    for k in np.argsort(at):
        if grp and at[k]-at[grp[-1][0]] <= 30: grp[-1].append(k)
        else: grp.append([k])
    gt = np.array([at[g[0]] for g in grp]); inside = (gt > 50) & (gt < len(y)/sr*1000-150)
    ra = nearest(ons, gt[inside]); hit = np.abs(ra) <= 30
    used = np.abs(nearest(gt[inside], ons)) <= 30
    # glove peaks vs MIDI (CSV timeline)
    raw = next((ROOT/'11-12的包/Data/Raw').glob(f'*_{RAWKEY}.csv'))
    cand = RD.L5.hit_cand.work((raw.stem, str(raw), 1.5, {}))[1]; ct = np.array(sorted(x['t'] for x in cand))
    s2, off2, rate2 = max(((s,)+tuple(AL.align(mt*s, ct)) for s in np.r_[1.0, np.arange(0.995, 1.005, 0.0005)]), key=lambda x: x[2])
    gcsv = np.array([mt[g[0]]*s2+off2 for g in grp]); rg = nearest(ct, gcsv); hg = np.abs(rg) <= 30
    # classification on single notes
    single = [g[0] for g in grp if len(g) == 1]
    single = [k for k in single if np.min(np.abs(np.delete(at, k)-at[k])) > 50 and 50 < at[k] < len(y)/sr*1000-150]
    X, Y, T = [], [], []
    for k in single:
        d = nearest(ons, np.array([at[k]]))[0]
        if abs(d) > 30: continue
        f = features(y, sr, at[k]+d)
        if f is not None: X.append(f); Y.append(ORDER.index(md[k])); T.append(at[k])
    X = np.array(X); Y = np.array(Y); T = np.array(T)
    import xgboost as xgb
    present = sorted(set(Y)); remap = {c: i for i, c in enumerate(present)}; Yr = np.array([remap[c] for c in Y])
    folds = np.array_split(np.argsort(T), 5); pred = np.zeros(len(Y), int)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(Y)), te)
        w = np.array([1.0/np.sum(Yr[tr] == c) for c in Yr[tr]]); w *= len(w)/w.sum()
        m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                              tree_method='hist', device='cuda', objective='multi:softprob', num_class=len(present))
        m.fit(X[tr], np.searchsorted(np.arange(len(present)), Yr[tr]), sample_weight=w)
        pred[te] = m.predict(X[te])
    P = np.array([present[i] for i in pred])
    acc = float(np.mean(P == Y)); conf = np.zeros((7, 7), int)
    for a, b in zip(Y, P): conf[a, b] += 1
    cym = {1, 2, 5}; fam = float(np.mean([(a in cym) == (b in cym) for a, b in zip(Y, P)]))
    res = dict(audio_s=len(y)/sr, onsets=len(ons), midi_notes=len(ev), audio_scale=s, audio_offset_ms=off, align_rate=rate,
               events=int(inside.sum()), recall=float(hit.mean()), precision=float(used.mean()),
               audio_jitter_ms=dict(median_abs=float(np.median(np.abs(ra[hit]))), p90=float(np.percentile(np.abs(ra[hit]), 90)), std=float(np.std(ra[hit]))),
               glove_jitter_ms=dict(median_abs=float(np.median(np.abs(rg[hg]))), p90=float(np.percentile(np.abs(rg[hg]), 90)), std=float(np.std(rg[hg])), matched=float(hg.mean())),
               singles=len(Y), class_counts={ORDER[c]: int(n) for c, n in sorted(Counter(Y.tolist()).items())},
               accuracy=acc, majority=float(max(Counter(Y.tolist()).values())/len(Y)), cymbal_vs_drum=fam,
               per_drum={ORDER[c]: [int(conf[c, c]), int(conf[c].sum())] for c in present}, confusion=conf.tolist())
    (OUT/'audio_1112.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    L = [f"audio {res['audio_s']:.1f} s, onsets {len(ons)}, MIDI notes {len(ev)} -> {res['events']} sounding events in the clip",
         f"audio<->MIDI: scale {s:.4f} offset {off:.0f} ms, match rate {rate:.3f}",
         f"onset detection: recall {res['recall']:.3f}  precision {res['precision']:.3f}",
         f"timing vs MIDI  audio: median |d| {res['audio_jitter_ms']['median_abs']:.1f} ms, p90 {res['audio_jitter_ms']['p90']:.1f}, std {res['audio_jitter_ms']['std']:.1f}",
         f"                glove: median |d| {res['glove_jitter_ms']['median_abs']:.1f} ms, p90 {res['glove_jitter_ms']['p90']:.1f}, std {res['glove_jitter_ms']['std']:.1f} (matched {res['glove_jitter_ms']['matched']:.3f})",
         f"drum from sound (single notes {len(Y)}, 5 time-block folds): accuracy {acc:.3f} (majority class {res['majority']:.3f}); cymbal vs drum {fam:.3f}"]
    L += [f"   {d:8s} {v[0]:3d}/{v[1]:3d}  -> " + ' '.join(f'{ORDER[j]}:{conf[ORDER.index(d), j]}' for j in range(7) if conf[ORDER.index(d), j] and j != ORDER.index(d)) for d, v in res['per_drum'].items()]
    txt = '\n'.join(L); print(txt); (OUT/'audio_1112.txt').write_text(txt+'\n', encoding='utf-8')
    plots(y, sr, env, hop, ons, at, md, conf, present, ra[hit], rg[hg])


def plots(y, sr, env, hop, ons, at, md, conf, present, ra, rg):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    col = dict(zip(ORDER, ['#d62728', '#1f77b4', '#9467bd', '#2ca02c', '#ff7f0e', '#17becf', '#8c564b']))
    t0 = at[np.searchsorted(np.sort(at), np.median(at))]; a, b = t0-2000, t0+2000
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.4, 0.8]))
    tt = np.arange(len(y))/sr*1000; m = (tt >= a) & (tt <= b)
    ax[0].plot(tt[m][::8]/1000, y[m][::8], lw=0.4, color='k'); ax[0].set_ylabel('waveform')
    i0, i1 = int(a/1000*sr), int(b/1000*sr)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y[i0:i1], n_fft=2048, hop_length=256)), ref=np.max)
    ax[1].imshow(S, origin='lower', aspect='auto', extent=[a/1000, b/1000, 0, sr/2/1000], cmap='magma', vmin=-70); ax[1].set_ylim(0, 16); ax[1].set_ylabel('kHz')
    te = np.arange(len(env))*hop/sr*1000; me = (te >= a) & (te <= b)
    ax[2].plot(te[me]/1000, env[me], lw=0.8, color='gray'); ax[2].set_ylabel('onset strength'); ax[2].set_xlabel('video time (s)')
    for t in ons[(ons >= a) & (ons <= b)]: ax[2].axvline(t/1000, color='k', lw=0.6, ls=':')
    for t, d in zip(at, md):
        if a <= t <= b:
            for k in (0, 2): ax[k].axvline(t/1000, color=col[d], lw=1.2, alpha=0.8)
    ax[0].legend(handles=[plt.Line2D([], [], color=col[d], lw=2, label=EN[d]) for d in ORDER], ncol=7, fontsize=8, loc='upper right')
    ax[0].set_title('11/12 song 1: video audio, MIDI notes (colour) and detected onsets (dotted)')
    fig.tight_layout(); fig.savefig(OUT/'audio_1112_excerpt.png', dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    c = conf[np.ix_(present, present)]; n = c/np.maximum(c.sum(1, keepdims=True), 1)
    ax[0].imshow(n, cmap='Blues', vmin=0, vmax=1); lab = [EN[ORDER[i]] for i in present]
    ax[0].set_xticks(range(len(lab)), lab, rotation=45); ax[0].set_yticks(range(len(lab)), lab); ax[0].set_xlabel('predicted from sound'); ax[0].set_ylabel('MIDI')
    for i in range(len(lab)):
        for j in range(len(lab)): ax[0].text(j, i, c[i, j], ha='center', va='center', fontsize=8, color='w' if n[i, j] > 0.5 else 'k')
    ax[0].set_title('drum from sound (single notes, time-block CV)')
    bins = np.arange(-30, 31, 2)
    ax[1].hist(ra, bins=bins, alpha=0.6, label=f'audio onset (std {np.std(ra):.1f} ms)'); ax[1].hist(rg, bins=bins, alpha=0.6, label=f'glove peak (std {np.std(rg):.1f} ms)')
    ax[1].set_xlabel('time minus MIDI trigger (ms, constant offset removed)'); ax[1].legend(); ax[1].set_title('timing spread vs electronic-drum MIDI')
    fig.tight_layout(); fig.savefig(OUT/'audio_1112_result.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
