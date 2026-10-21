"""10/23 runtime: venue-free committee + causal per-drum offset correction + cymbal light-hit rescue.

Per hit (released by the detector at peak + 80 ms):
  1. features: heading-free trajectory summary + rotation relative to the recorded drumhead zero
     (same 104 dims as 10/20, see venue_free_runtime_1020);
  2. per-drum offset correction: from EARLIER hits of the same hand in this take that the model was
     confident about (score >= 0.7), compare their mean rotation per predicted drum with that drum's
     training mean; the count-weighted gap shifts this hit's rotation block (after >= 20 such hits).
     Done twice (second pass uses the corrected predictions of earlier hits). No labels, no future;
  3. cymbal light-hit rescue: the detector runs at 1.7 g; peaks below 2.1 g are reported only when
     the model calls them Hi-Hat / Crash / Ride with score >= 0.6. All peaks feed the history.
"""
from collections import deque

import numpy as np

import venue_free_runtime_1020 as base

CONTRACT = 'hologrip_venue_free_committee_1023_v1'
INPUT_DIMENSIONS = base.INPUT_DIMENSIONS
POSE_DTS = base.POSE_DTS
ROT = slice(83, 104)
feature = base.feature
probabilities = base.probabilities
attach_boosters = base.attach_boosters


class _Stats:
    """Running per-drum sums of raw rotation blocks of confident hits (one per pass)."""
    def __init__(self):
        self.sum = np.zeros((7, 21)); self.n = np.zeros(7); self.total = 0

    def add(self, rot, p, conf):
        if p.max() >= conf:
            d = int(p.argmax()); self.sum[d] += rot; self.n[d] += 1; self.total += 1

    def offset(self, tm):
        m = self.n > 0
        return (self.n[m, None]*tm[m]-self.sum[m]).sum(0)/self.n[m].sum()


class StreamEngine:
    def __init__(self, bundle):
        self.bundle = bundle; ad = bundle['adaptation']; cr = bundle['cymbal_rescue']
        self.alpha, self.conf, self.minh = ad['alpha'], ad['confidence'], ad['min_confident_hits']
        self.strong_g, self.cym, self.min_score = cr['strong_g'], set(cr['cymbals']), cr['min_score']
        self.classes = [str(c) for c in bundle['classes']]
        self.tm = {h: np.array([bundle['drum_rotation_means'].get(f'{h}|{d}', np.zeros(21)) for d in self.classes]) for h in 'LR'}
        self.history = {h: deque(maxlen=1024) for h in 'LR'}
        self.hit = {h: base.world.base.detector.HitStream(h, bundle['detector_config']) for h in 'LR'}
        self.stats = {h: [_Stats(), _Stats()] for h in 'LR'}
        self.hand_mode = {h: None for h in 'LR'}; self.zero_yaw = {h: None for h in 'LR'}; self.zero_pitch = {h: None for h in 'LR'}
        self.heading_distance = {h: None for h in 'LR'}; self.count = 0; self.adapted = {h: 0 for h in 'LR'}

    def _classify(self, hand, x):
        st = self.stats[hand]; p0 = probabilities(self.bundle, x[None, :], hand)[0]; p = p0; x1 = x
        if st[0].total >= self.minh:
            x1 = x.copy(); x1[ROT] = x[ROT]+self.alpha*st[0].offset(self.tm[hand]); p = probabilities(self.bundle, x1[None, :], hand)[0]
        p1 = p; pf = p1
        if st[1].total >= self.minh:
            x2 = x.copy(); x2[ROT] = x[ROT]+self.alpha*st[1].offset(self.tm[hand]); pf = probabilities(self.bundle, x2[None, :], hand)[0]
            self.adapted[hand] += 1
        st[0].add(x[ROT], p0, self.conf); st[1].add(x[ROT], p1, self.conf)
        return pf

    def push(self, hand, sample):
        self.zero_yaw[hand] = (sample['absolute_yaw']-sample['yaw']+180) % 360-180
        self.zero_pitch[hand] = sample['absolute_pitch']-sample['pitch']
        self.hand_mode[hand] = 'venue_free_adaptive'
        self.history[hand].append(sample)
        peaks = self.hit[hand].push(sample)
        if not peaks: return []
        rows = list(self.history[hand])
        prepared = base.world.base.legacy.prepare({hand: rows}); first = rows[0]['index']; output = []
        for peak in peaks:
            local = {**peak, 'peak_index': int(peak['peak_index'])-first, 'available_index': len(rows)-1}
            vector = feature(prepared, local, self.zero_yaw[hand], self.zero_pitch[hand])
            if vector is None: continue
            q = self._classify(hand, vector); top = np.sort(q); drum = self.classes[int(q.argmax())]
            if float(peak.get('peak_g', 99)) < self.strong_g and not (drum in self.cym and q.max() >= self.min_score):
                continue
            self.count += 1
            output.append(dict(**{k: v for k, v in peak.items() if k != 'feature'}, prediction_id=self.count,
                               direction_mode='venue_free_adaptive', zero_yaw_deg=self.zero_yaw[hand], heading_distance_deg=None,
                               drum=drum, model_score=float(q.max()), model_margin=float(top[-1]-top[-2])))
        return output
