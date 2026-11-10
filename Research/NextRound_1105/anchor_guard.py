"""Anchor safeguard (research): revoke a wrong automatic anchor using later hits only (causal).

Base engine: at the N-th hit the first N hits may anchor the offset statistics on one drum (snare, or a
drum the committee consistently names). Once anchored, the decision is never revisited, so a wrong anchor
(e.g. opening snare taps read as Hi-Hat, 10/27 'ton2 L') shifts the whole take by one drum.

GuardEngine keeps a shadow copy of the offset statistics built as if no anchor had been taken (the first N
hits enter by confidence like every other hit). After the anchor, for every hit where the shadow statistics
are ready (>= min_confident_hits confident hits), it compares the committee's log max probability with the
anchored correction (a) and with the shadow correction (b). When at least `after` such hits are counted and
mean(b - a) > `margin`, the anchor is revoked: the shadow statistics replace the anchored ones (exactly the
state the engine would be in had it never anchored) and the hit is classified with them. Earlier outputs
are not changed. Revocation happens at most once per hand; afterwards the engine runs unanchored.

Activation (research harness): HG_GUARD='{"after": 15, "margin": 0.05}' or install(after=..., margin=...).
"""
from __future__ import annotations
import json, os

import numpy as np

GUARD = dict(after=15, margin=0.05)


def install(runtime, **params):
    G = dict(GUARD, **params)
    base = runtime.StreamEngine
    if getattr(base, '_guard', False): base = base.__mro__[1]
    _Stats, ROT, probabilities, _yaw_pitch = runtime._Stats, runtime.ROT, runtime.probabilities, runtime._yaw_pitch

    def _copy(st):
        c = _Stats(); c.sum = st.sum.copy(); c.n = st.n.copy(); c.total = st.total; return c

    class GuardEngine(base):
        _guard = True
        guard_params = G

        def __init__(self, bundle, snare_anchor=0):
            super().__init__(bundle, snare_anchor)
            self.shadow = {h: [_Stats(), _Stats()] for h in 'LR'}
            self.gsum = {h: [0.0, 0.0, 0] for h in 'LR'}
            self.revoked = {h: False for h in 'LR'}; self.revoked_at = {h: None for h in 'LR'}
            self.revoked_drum = {h: None for h in 'LR'}

        def _replay(self, hand, X, t):
            if self.revoked[hand]:
                a = self.anchor; self.anchor = 0
                try: super()._replay(hand, X, t)
                finally: self.anchor = a
                self.anchored[hand] = False; self.anchored_drum[hand] = None
                return
            a = self.anchor; self.anchor = 0
            try: super()._replay(hand, X, t)
            finally: self.anchor = a
            self.shadow[hand] = [_copy(s) for s in self.stats[hand]]
            super()._replay(hand, X, t)
            self.gsum[hand] = [0.0, 0.0, 0]

        def _classify(self, hand, x):
            st = self.stats[hand]; holding = self.accepted[hand] < self.anchor
            ready = [s.total >= self.minh or (self.anchored[hand] is True and s.n[self.anchored_drum[hand]] >= self.anchor) for s in st]
            use1, use2 = ready
            guard = self.anchor and not self.revoked[hand] and (holding or self.anchored[hand] is True)
            sh = self.shadow[hand]; s1 = s2 = False
            rows = [x]; x2 = x
            if use1:
                x1 = x.copy(); x1[ROT] = x[ROT]+self.alpha*st[0].offset(self.tm[hand]); rows.append(x1)
            if use2:
                x2 = x.copy(); x2[ROT] = x[ROT]+self.alpha*st[1].offset(self.tm[hand]); rows.append(x2)
                self.adapted[hand] += 1
            if guard and not holding:
                s1 = sh[0].total >= self.minh; s2 = sh[1].total >= self.minh
                if s1: xs1 = x.copy(); xs1[ROT] = x[ROT]+self.alpha*sh[0].offset(self.tm[hand]); rows.append(xs1)
                if s2: xs2 = x.copy(); xs2[ROT] = x[ROT]+self.alpha*sh[1].offset(self.tm[hand]); rows.append(xs2)
            P = probabilities(self.bundle, np.stack(rows), hand); i = 1
            p0 = P[0]
            p1 = P[i] if use1 else p0; i += use1
            pf = P[i] if use2 else p1; i += use2
            ps1 = P[i] if s1 else p0; i += s1
            psf = P[i] if s2 else ps1
            self._last_p0 = p0
            if guard:
                sh[0].add(x[ROT], p0, self.conf); sh[1].add(x[ROT], ps1 if not holding else p0, self.conf)
            if holding:
                self.pending[hand].append((x[ROT].copy(), p0, p1, _yaw_pitch(x)))
                if len(self.pending[hand]) == self.anchor: self._settle_anchor(hand)
            else:
                st[0].add(x[ROT], p0, self.conf); st[1].add(x[ROT], p1, self.conf)
                if guard and s2:
                    g = self.gsum[hand]; g[0] += float(np.log(pf.max()+1e-9)); g[1] += float(np.log(psf.max()+1e-9)); g[2] += 1
                    if g[2] >= G['after'] and (g[1]-g[0])/g[2] > G['margin']:
                        self.revoked[hand] = True; self.revoked_at[hand] = self.accepted[hand]+1
                        self.revoked_drum[hand] = self.classes[self.anchored_drum[hand]]
                        self.stats[hand] = [_copy(s) for s in sh]
                        self.anchored[hand] = False; self.anchored_drum[hand] = None
                        pf = psf; x2 = xs2
            self.accepted[hand] += 1
            yp = _yaw_pitch(x2)
            return self._position_prior(hand, pf, yp), yp

    runtime.StreamEngine = GuardEngine
    return GuardEngine


def install_from_env(runtime):
    v = os.environ.get('HG_GUARD', '')
    if v and v != 'off': return install(runtime, **json.loads(v))
    return None


STATS = dict(min_drums=0, per_drum=5, cap_deg=0.0, noadapt=False, balance=0, min_n=0, power=1.0)


def install_stats(runtime, **params):
    """Research variants of the offset statistics (env HG_STATS):
      min_drums/per_drum : the confident-hit count only makes the correction ready once >= min_drums drums each
                           hold >= per_drum confident hits (a single-drum stretch cannot set the offset alone)
      cap_deg            : each of the 7 pose offsets is shrunk to at most cap_deg degrees
      noadapt            : no offset correction at all (anchor off too)
      balance            : each drum counts as at most `balance` hits in the offset average (a long single-drum
                           stretch cannot outvote the other drums)
      min_n              : drums with fewer than min_n confident hits are left out of the offset average
      power              : each drum's weight is n**power instead of n (0.5 = square root)"""
    S = dict(STATS, **params)
    base = runtime._Stats
    if getattr(base, '_variant', False): base = base.__mro__[1]

    class Stats(base):
        _variant = True
        params = S

        def __init__(self):
            super().__init__(); self._t = 0

        @property
        def total(self):
            if S['noadapt']: return 0
            if S['min_drums'] and int((self.n >= S['per_drum']).sum()) < S['min_drums']: return 0
            return self._t

        @total.setter
        def total(self, v): self._t = v

        def offset(self, tm):
            if S['balance'] or S['min_n'] or S['power'] != 1.0:
                m = self.n >= max(1, S['min_n'])
                if not m.any(): m = self.n > 0
                c = np.minimum(self.n[m], S['balance']) if S['balance'] else self.n[m]
                w = c**S['power']/self.n[m]
                o = (w[:, None]*(self.n[m, None]*tm[m]-self.sum[m])).sum(0)/(w*self.n[m]).sum()
            else:
                o = super().offset(tm)
            if S['cap_deg']:
                b = o.reshape(7, 3); deg = np.linalg.norm(b, axis=1)*180
                f = np.minimum(1.0, S['cap_deg']/np.maximum(deg, 1e-9)); o = (b*f[:, None]).reshape(-1)
            return o

    runtime._Stats = Stats
    if S['noadapt']:
        se = runtime.StreamEngine; init = se.__init__
        def __init__(self, bundle, snare_anchor=0): init(self, bundle, 0)
        se.__init__ = __init__
    return Stats


def install_stats_from_env(runtime):
    v = os.environ.get('HG_STATS', '')
    if v and v != 'off': return install_stats(runtime, **json.loads(v))
    return None
