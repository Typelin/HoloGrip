"""Remove electronic-drum double triggers from a MIDI answer key.

Two notes of the same drum within 150 ms are one physical stroke when the gloves show only one stroke
for them: only one of the notes has a glove peak (>= 1.5 g, either hand, within +-40 ms), or both
notes point to the same peak. The note with the peak is kept. Two separate peaks (two real strokes,
e.g. left/right alternation at 55 ms) keep both notes. Uses glove peaks only, never model output.
"""
from __future__ import annotations
import numpy as np

GAP_MS, PEAK_MS, PEAK_G = 150.0, 40.0, 1.5


def peak_of(t, cand):
    best = None
    for k, c in enumerate(cand):
        d = abs(c['t']-t)
        if c['g'] >= PEAK_G and d <= PEAK_MS and (best is None or d < best[0]): best = (d, k)
    return None if best is None else best[1]


def dedupe(truth, cand):
    T = sorted(truth, key=lambda g: g['t']); drop = set()
    for i, a in enumerate(T):
        if i in drop: continue
        for j in range(i+1, len(T)):
            b = T[j]
            if b['t']-a['t'] > GAP_MS: break
            if j in drop or b['drum'] != a['drum']: continue
            pa, pb = peak_of(a['t'], cand), peak_of(b['t'], cand)
            if pb is None or pa == pb: drop.add(j)
            elif pa is None: drop.add(i); break
    return [g for k, g in enumerate(T) if k not in drop], len(drop)
