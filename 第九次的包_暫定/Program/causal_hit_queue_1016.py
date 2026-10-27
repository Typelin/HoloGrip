"""Queued causal hit events with optional weak opposite-hand motion suppression.

Detection and feature collection are separate: two confirmed hits may wait
for their own +80ms contexts at the same time. Cross-hand suppression sees
only opposite peaks already confirmed before the current emission packet.
"""
from collections import deque

import numpy as np
import causal_hit_detector_1016 as shapes

rt = shapes.rt
CONTRACT = 'hologrip_causal_queued_hit_v2'


class QueueHand:
    def __init__(self, hand, config):
        self.hand, self.cfg = hand, config
        self.buf = deque(maxlen=512)
        self.waiting = []
        self.last = None
        self.observed = deque()

    def push(self, sample):
        self.buf.append(sample)
        recent = list(self.buf)[-14:]
        n = self.cfg['confirm_samples']
        if len(recent) >= max(8, 2*n+1):
            j = len(recent)-n-1
            c = recent[j]
            if c['mag'] > max(s['mag'] for s in recent[j-n:j]) and c['mag'] >= max(s['mag'] for s in recent[j+1:j+n+1]):
                prior = recent[max(0, j-6):j+1]
                rise = c['mag']-min(s['mag'] for s in prior)
                snap = max([0.] + [b['mag']-a['mag'] for a, b in zip(prior, prior[1:])][-3:])
                if c['mag'] >= self.cfg['threshold_g'] and rise >= self.cfg['rise_g'] and snap >= self.cfg['snap_g']:
                    self.observed.append((c, sample['row_index']))
                    if self.last is None or c['t']-self.last['t'] >= self.cfg['refractory_ms']:
                        self.waiting.append(c)
                        self.last = c
                    elif self.waiting and self.waiting[-1] is self.last and c['mag'] > self.last['mag']*1.25:
                        self.waiting[-1] = c
                        self.last = c
        while self.observed and self.observed[0][0]['t'] < sample['t']-500:
            self.observed.popleft()
        output = []
        while self.waiting and sample['t'] >= self.waiting[0]['t']+80:
            c = self.waiting.pop(0)
            f = rt.feature(list(self.buf), c['t'], self.hand, 80)
            if f is not None:
                output.append(dict(hand=self.hand, peak_ms=c['t'], emitted_ms=sample['t'], peak_g=c['mag'],
                    peak_index=c['index'], available_index=sample['index'], available_row=sample['row_index'],
                    feature=f.tolist(), window_samples=sum(abs(s['t']-c['t']) <= 80 for s in self.buf)))
        return output


class JointStream:
    def __init__(self, config):
        self.cfg = config
        self.hands = {h: QueueHand(h, config) for h in 'LR'}
        self.suppressed = 0

    def push(self, hand, sample):
        own = self.hands[hand]
        possible = own.push(sample)
        output = []
        for p in possible:
            peers = self.hands['R' if hand == 'L' else 'L'].observed
            shadow = (self.cfg['contrast_ratio'] > 0 and p['peak_g'] <= self.cfg['weak_ceiling_g'] and
                any(abs(q['t']-p['peak_ms']) <= self.cfg['contrast_ms'] and
                    p['peak_g']-1 < self.cfg['contrast_ratio']*max(.01, q['mag']-1)
                    for q, arrival in peers if arrival <= sample['row_index']))
            if shadow:
                self.suppressed += 1
            else:
                output.append(p)
        return output


def detect(hands, config):
    stream = JointStream(config)
    all_rows = sorted(((s['row_index'], h, s) for h, rows in hands.items() for s in rows), key=lambda x: x[0])
    output = []
    for _, h, s in all_rows:
        output.extend(stream.push(h, s))
    output.sort(key=lambda p: (p['peak_ms'], p['hand']))
    for i, p in enumerate(output):
        p['prediction_id'] = i+1
    return output, dict(opposite_hand_weak_suppressed=stream.suppressed,
                        pending_tail={h: len(s.waiting) for h, s in stream.hands.items()})


def fast_own(cache, config):
    ok = ((cache['mags'] >= config['threshold_g']) & (cache['rise'] >= config['rise_g']) &
          (cache['snap'] >= config['snap_g']))
    indices = np.flatnonzero(ok)
    ts = cache['sample_times']
    accepted, waiting, last = [], [], None

    def emit(k, when):
        c = float(cache['times'][k])
        left = np.searchsorted(ts, c-80, side='left')
        right = min(np.searchsorted(ts, c+80, side='right'), when+1)
        if right-left >= 6:
            s = cache['samples'][int(cache['indices'][k])]
            accepted.append(dict(peak_ms=c, emitted_ms=float(ts[when]), peak_g=float(cache['mags'][k]),
                peak_index=s['index'], available_index=int(when), available_row=cache['samples'][when]['row_index']))

    for k in indices:
        while waiting and waiting[0][1] < cache['confirm'][k]:
            q, when = waiting.pop(0)
            if when < len(ts):
                emit(q, when)
        c = float(cache['times'][k])
        ready = max(int(cache['confirm'][k]), int(np.searchsorted(ts, c+80)))
        if last is None or c-cache['times'][last] >= config['refractory_ms']:
            waiting.append((k, ready)); last = k
        elif waiting and waiting[-1][0] == last and cache['mags'][k] > cache['mags'][last]*1.25:
            waiting[-1] = (k, ready); last = k
        while waiting and waiting[0][1] <= cache['confirm'][k]:
            q, when = waiting.pop(0)
            if when < len(ts):
                emit(q, when)
    for q, when in waiting:
        if when < len(ts):
            emit(q, when)
    potentials = [dict(t=float(cache['times'][k]), mag=float(cache['mags'][k]),
                       arrival=cache['samples'][int(cache['confirm'][k])]['row_index']) for k in indices]
    return accepted, potentials


def fast_detect(caches, config):
    output, potentials = {}, {}
    for h in 'LR':
        output[h], potentials[h] = fast_own(caches[h][config['confirm_samples']], config)
    combined = []
    for h in 'LR':
        peers = potentials['R' if h == 'L' else 'L']
        times = np.asarray([q['t'] for q in peers])
        for p in output[h]:
            shadow = False
            if config['contrast_ratio'] and p['peak_g'] <= config['weak_ceiling_g']:
                a = np.searchsorted(times, p['peak_ms']-config['contrast_ms'])
                b = np.searchsorted(times, p['peak_ms']+config['contrast_ms'], side='right')
                shadow = any(q['arrival'] <= p['available_row'] and
                    p['peak_g']-1 < config['contrast_ratio']*max(.01, q['mag']-1) for q in peers[a:b])
            if not shadow:
                combined.append(dict(hand=h, **p))
    combined.sort(key=lambda p: (p['peak_ms'], p['hand']))
    for i, p in enumerate(combined):
        p['prediction_id'] = i+1
    return combined
