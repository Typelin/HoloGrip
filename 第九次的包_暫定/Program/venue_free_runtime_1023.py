"""9th package runtime (file/contract version tag 1023 kept): hit gate + weighted venue-free committee + causal per-drum offset correction.

Committee: members trained on raw rows and members trained on rows re-centred per recording and hand;
their probabilities are averaged with bundle['member_weights'] (equal weights when absent).
Position prior (bundle['position_prior']): the final drum probabilities are multiplied by
exp(-d^2 / 2 sigma^2)^weight, d = distance between this hit's (corrected) yaw/pitch and each drum's
training position (circular yaw), then renormalised. Applied to every hit before pose continuity.

Per candidate peak (detector at 1.5 g, released at peak + 80 ms):
  1. hit gate: a learned classifier decides whether the stick really hit a drum, from the peak's
     shape (detector features), the 104 venue-free features, the other hand's motion within +-40 ms
     and the time / strength relative to the previous candidate of the same hand. Non-hits are dropped
     and do not enter any history.
  2. drum: venue-free committee with per-drum offset correction from EARLIER accepted hits of the same
     hand in this take (score >= bundle confidence, after >= 20 such hits, two passes). No labels, no future.
  3. pose continuity: if this hit's score is < 0.9, the same hand's previous accepted hit was within
     600 ms with a zero-relative yaw/pitch within 16 deg, and that previous drum is among this hit's two
     highest-scoring drums, it is the same drum as that previous hit.
Automatic snare anchor (snare_anchor=N): the first N accepted hits of each hand are held back from the
offset statistics. At the N-th hit they are used as a snare anchor only if, for every one of them, the
uncorrected committee ranks the snare first and all N zero-relative poses lie within 8 deg of
their median (the player tapped one drum that looks like the snare). Then they enter the statistics as
snare and the correction starts right away. If they are not snare-like but still one tight spot that
the committee calls the same drum every time (mean score >= 0.7), they anchor on that drum instead
(e.g. a take that opens on the Hi-Hat). Otherwise they enter normally and the correction starts
after 20 confident hits. No protocol is assumed and no MIDI is used. The model's own output for those
hits is unchanged.
"""
from scipy.spatial.transform import Rotation as _Rot
from collections import deque

import numpy as np

import venue_free_runtime_1020 as base

CONTRACT = 'hologrip_venue_free_committee_1023_v1'
INPUT_DIMENSIONS = base.INPUT_DIMENSIONS
POSE_DTS = base.POSE_DTS
ROT = slice(83, 104)
feature = base.feature


def probabilities(bundle, X, hand):
    """Weighted committee average of (XGBoost + MLP)/2; columns follow bundle['classes']."""
    X = np.asarray(X, dtype=np.float32); weights = bundle.get('member_weights') or {k: 1.0 for k in bundle['portable']}
    out = 0
    for key, member in bundle['portable'].items():
        px = bundle['_boosters'][key, hand].inplace_predict(X)
        out = out+weights[key]*(px+base.mlp_proba(member[hand]['mlp'], X))/2
    return out/sum(weights[k] for k in bundle['portable'])


CONTINUITY = dict(max_ms=600.0, max_deg=16.0, below_score=0.9, previous_in_top=2)
PRETEST_SNARE_ANCHOR = 5
ANCHOR_DRUM = '小鼓'
ANCHOR_RULE = dict(snare_in_top=1, max_pose_deg=8.0, any_drum_min_score=0.7)


def _yaw_pitch(x):
    v = np.asarray(x[ROT]).reshape(7, 3)[4]*np.pi
    return _Rot.from_rotvec(v).as_euler('ZYX', degrees=True)[:2]


def attach_boosters(bundle):
    base.attach_boosters(bundle)
    gate = bundle['hit_gate']; bst = bundle['_xgb'].Booster(); bst.load_model(bytearray(gate['xgb_json'])); bst.set_param({'nthread': 1})
    bundle['_hit_gate'] = bst
    return bundle


def hit_probability(bundle, g, det, x, ctx):
    d = np.asarray(det, np.float32) if len(det) else np.zeros(18, np.float32)
    v = np.r_[g, d, x, ctx].astype(np.float32)[None, :]
    return float(bundle['_hit_gate'].inplace_predict(v)[0])


class _Stats:
    """Running per-drum sums of raw rotation blocks of confident hits (one per pass)."""
    def __init__(self):
        self.sum = np.zeros((7, 21)); self.n = np.zeros(7); self.total = 0

    def add(self, rot, p, conf):
        if p.max() >= conf:
            self.force(rot, int(p.argmax()))

    def force(self, rot, d):
        self.sum[d] += rot; self.n[d] += 1; self.total += 1

    def offset(self, tm):
        m = self.n > 0
        return (self.n[m, None]*tm[m]-self.sum[m]).sum(0)/self.n[m].sum()


class StreamEngine:
    def __init__(self, bundle, snare_anchor=0):
        self.bundle = bundle; ad = bundle['adaptation']; self.anchor = int(snare_anchor)
        self.alpha, self.conf, self.minh = ad['alpha'], ad['confidence'], ad['min_confident_hits']
        self.gate_threshold = bundle['hit_gate']['threshold']
        self.classes = [str(c) for c in bundle['classes']]
        self.tm = {h: np.array([bundle['drum_rotation_means'].get(f'{h}|{d}', np.zeros(21)) for d in self.classes]) for h in 'LR'}
        self.history = {h: deque(maxlen=1024) for h in 'LR'}
        self.hit = {h: base.world.base.detector.HitStream(h, bundle['detector_config']) for h in 'LR'}
        self.stats = {h: [_Stats(), _Stats()] for h in 'LR'}
        self.last_candidate = {h: None for h in 'LR'}
        self.hand_mode = {h: None for h in 'LR'}; self.zero_yaw = {h: None for h in 'LR'}; self.zero_pitch = {h: None for h in 'LR'}
        self.heading_distance = {h: None for h in 'LR'}; self.count = 0; self.adapted = {h: 0 for h in 'LR'}
        self.rejected = {h: 0 for h in 'LR'}; self.previous = {h: None for h in 'LR'}; self.continued = {h: 0 for h in 'LR'}
        self.anchor_drum = self.classes.index(ANCHOR_DRUM); self.accepted = {h: 0 for h in 'LR'}
        self.pending = {h: [] for h in 'LR'}; self.anchored = {h: None for h in 'LR'}; self.anchored_drum = {h: None for h in 'LR'}
        self.position = bundle.get('position_prior')
        self.tm_pose = {h: np.array([_yaw_pitch(np.r_[np.zeros(83), m]) for m in self.tm[h]]) for h in 'LR'}

    def _position_prior(self, hand, p, yp):
        """Down-weight drums whose training position (yaw/pitch at the hit, circular yaw) is far from this hit."""
        if not self.position: return p
        d = yp[None, :]-self.tm_pose[hand]; d[:, 0] = (d[:, 0]+180) % 360-180
        q = p*np.exp(-(d**2).sum(1)/(2*self.position['sigma_deg']**2))**self.position['weight']+1e-12
        return q/q.sum()

    def _settle_anchor(self, hand):
        """Decide the automatic snare anchor from the first N accepted hits (all already emitted)."""
        st = self.stats[hand]; pend = self.pending[hand]; self.pending[hand] = []
        poses = np.array([yp for _, _, _, yp in pend]); med = np.median(poses, 0)
        tight = bool(np.all(np.hypot(*(poses-med).T) <= ANCHOR_RULE['max_pose_deg']))
        drum = None
        if tight and all(self.anchor_drum in np.argsort(-p0)[:ANCHOR_RULE['snare_in_top']] for _, p0, _, _ in pend):
            drum = self.anchor_drum
        elif tight:
            # one tight spot that the committee consistently calls the same drum: anchor on that drum
            P0 = np.array([p0 for _, p0, _, _ in pend]); top = P0.argmax(1)
            if np.all(top == top[0]) and P0[:, top[0]].mean() >= ANCHOR_RULE['any_drum_min_score']: drum = int(top[0])
        self.anchored[hand] = drum is not None; self.anchored_drum[hand] = drum
        for rot, p0, p1, _ in pend:
            if drum is not None: st[0].force(rot, drum); st[1].force(rot, drum)
            else: st[0].add(rot, p0, self.conf); st[1].add(rot, p1, self.conf)

    def _classify(self, hand, x):
        # Both corrections depend only on earlier hits, so all inputs go to the committee in one batch.
        st = self.stats[hand]; holding = self.accepted[hand] < self.anchor
        ready = [s.total >= self.minh or (self.anchored[hand] is True and s.n[self.anchored_drum[hand]] >= self.anchor) for s in st]
        use1, use2 = ready
        rows = [x]; x2 = x
        if use1:
            x1 = x.copy(); x1[ROT] = x[ROT]+self.alpha*st[0].offset(self.tm[hand]); rows.append(x1)
        if use2:
            x2 = x.copy(); x2[ROT] = x[ROT]+self.alpha*st[1].offset(self.tm[hand]); rows.append(x2)
            self.adapted[hand] += 1
        P = probabilities(self.bundle, np.stack(rows), hand)
        p0 = P[0]; p1 = P[1] if use1 else p0; pf = P[-1] if use2 else p1
        if holding:
            self.pending[hand].append((x[ROT].copy(), p0, p1, _yaw_pitch(x)))
            if len(self.pending[hand]) == self.anchor: self._settle_anchor(hand)
        else:
            st[0].add(x[ROT], p0, self.conf); st[1].add(x[ROT], p1, self.conf)
        self.accepted[hand] += 1
        yp = _yaw_pitch(x2)
        return self._position_prior(hand, pf, yp), yp

    def _other_peak(self, hand, t):
        other = 'R' if hand == 'L' else 'L'
        vals = [s['mag'] for s in self.history[other] if abs(s['t']-t) <= 40]
        return max(vals) if vals else 1.0

    def push(self, hand, sample):
        self.zero_yaw[hand] = (sample['absolute_yaw']-sample['yaw']+180) % 360-180
        self.zero_pitch[hand] = sample['absolute_pitch']-sample['pitch']
        self.hand_mode[hand] = 'venue_free_adaptive'
        self.history[hand].append(sample)
        peaks = self.hit[hand].push(sample)
        if not peaks: return []
        rows = list(self.history[hand])
        prepared = base.world.base.legacy.prepare({hand: rows}); first = rows[0]['index']; output = []
        for peak in sorted(peaks, key=lambda p: p['peak_ms']):
            local = {**peak, 'peak_index': int(peak['peak_index'])-first, 'available_index': len(rows)-1}
            vector = feature(prepared, local, self.zero_yaw[hand], self.zero_pitch[hand])
            if vector is None: continue
            t, g = float(peak['peak_ms']), float(peak['peak_g']); pv = self.last_candidate[hand]
            ctx = np.array([self._other_peak(hand, t), np.log1p(min(t-pv[0], 3000)) if pv else np.log1p(3000), g/pv[1] if pv else 1.0], np.float32)
            self.last_candidate[hand] = (t, g)
            ph = hit_probability(self.bundle, g, peak.get('feature', []), vector, ctx)
            if ph < self.gate_threshold:
                self.rejected[hand] += 1; continue
            q, yp = self._classify(hand, vector); top = np.sort(q); label = int(q.argmax()); prev = self.previous[hand]
            if (prev is not None and q.max() < CONTINUITY['below_score'] and t-prev[0] <= CONTINUITY['max_ms']
                    and prev[2] in np.argsort(-q)[:CONTINUITY['previous_in_top']]
                    and float(np.hypot(*(yp-prev[1]))) <= CONTINUITY['max_deg']):
                label = prev[2]; self.continued[hand] += 1
            self.previous[hand] = (t, yp, label)
            self.count += 1
            output.append(dict(**{k: v for k, v in peak.items() if k != 'feature'}, prediction_id=self.count,
                               direction_mode='venue_free_adaptive', zero_yaw_deg=self.zero_yaw[hand], heading_distance_deg=None,
                               drum=self.classes[label], model_score=float(q.max()), model_margin=float(top[-1]-top[-2]),
                               hit_probability=ph))
        return output
