"""Causal external IMU hit detector; no MIDI, labels or classifier required.

Parameters are selected separately, not hardcoded from per-event answers.
The reference stream emits at most once for a pending peak. All decisions use
packets already received, including when PC timestamps are identical.
"""
from collections import deque
import argparse
import json
from pathlib import Path

import numpy as np
import fourth_song03_raw_runtime_1008 as rt

CONTRACT = 'hologrip_causal_shape_hit_v1'


class HitStream:
    def __init__(self, hand, config):
        self.hand, self.config = hand, dict(config)
        self.buf = deque(maxlen=512)
        self.pending = None
        self.last_t = -1e12
        self.last_mag = 0.
        self.suppressed = self.replaced = 0

    def push(self, sample):
        self.buf.append(sample)
        n = self.config['confirm_samples']
        if len(self.buf) >= max(8, 2*n+1):
            # At most fourteen recent packets are needed for peak shape.
            recent = list(self.buf)[-max(14, 2*n+1):]
            j = len(recent)-n-1
            c = recent[j]
            left, right = recent[j-n:j], recent[j+1:j+n+1]
            if c['mag'] > max(s['mag'] for s in left) and c['mag'] >= max(s['mag'] for s in right):
                prior = recent[max(0, j-6):j+1]
                delta = [b['mag']-a['mag'] for a, b in zip(prior, prior[1:])]
                snap = max([0.] + delta[-3:])
                rise = c['mag']-min(s['mag'] for s in prior)
                cfg = self.config
                qualifies = c['mag'] >= cfg['threshold_g'] and rise >= cfg['rise_g'] and snap >= cfg['snap_g']
                if qualifies:
                    if c['t']-self.last_t < cfg['refractory_ms']:
                        self.suppressed += 1
                    elif self.pending is None:
                        self.pending = c
                    elif c['t']-self.pending['t'] < cfg['merge_ms']:
                        if c['mag'] > self.pending['mag']:
                            self.pending = c
                            self.replaced += 1
                        else:
                            self.suppressed += 1
        if self.pending is None or sample['t'] < self.pending['t']+self.config['context_ms']:
            return []
        peak = self.pending
        self.pending = None
        self.last_t, self.last_mag = peak['t'], peak['mag']
        # Feature context stays compatible with the unchanged drum models.
        values = rt.feature(list(self.buf), peak['t'], self.hand, 80)
        if values is None:
            return []
        return [dict(hand=self.hand, peak_ms=peak['t'], emitted_ms=sample['t'],
                     peak_g=peak['mag'], peak_index=peak['index'], available_index=sample['index'],
                     feature=values.tolist(), window_samples=sum(abs(s['t']-peak['t']) <= 80 for s in self.buf))]


def detect(hands, config):
    out, diag = [], {}
    for hand, rows in hands.items():
        stream = HitStream(hand, config)
        for row in rows:
            out.extend(stream.push(row))
        diag[hand] = dict(rows=len(rows), replaced=stream.replaced, suppressed=stream.suppressed,
                          incomplete_tail=int(stream.pending is not None))
    out.sort(key=lambda p: (p['peak_ms'], p['hand']))
    for i, p in enumerate(out):
        p['prediction_id'] = i+1
    return out, diag


def geometry(samples, confirm):
    """Search cache: every shape measurement uses only the peak and its past."""
    if len(samples) < max(8, 2*confirm+1):
        return dict(indices=np.asarray([], dtype=int), times=np.asarray([]), confirm=np.asarray([], dtype=int),
                    mags=np.asarray([]), rise=np.asarray([]), snap=np.asarray([]), sample_times=np.asarray([]), samples=samples)
    mags = np.asarray([s['mag'] for s in samples])
    times = np.asarray([s['t'] for s in samples])
    n = confirm
    centers = np.arange(max(7-n, n), len(samples)-n)
    ok = np.ones(len(centers), dtype=bool)
    for d in range(1, n+1):
        ok &= (mags[centers] > mags[centers-d]) & (mags[centers] >= mags[centers+d])
    indices = centers[ok]
    rise = mags[indices]-np.min(np.column_stack([mags[np.maximum(0, indices-k)] for k in range(7)]), axis=1)
    snap = np.maximum(0., np.max(np.column_stack([mags[np.maximum(0, indices-k)]-mags[np.maximum(0, indices-k-1)] for k in range(3)]), axis=1))
    return dict(indices=indices, times=times[indices], confirm=indices+n, mags=mags[indices],
                rise=rise, snap=snap, sample_times=times, samples=samples)


def fast_detect_one(cache, config):
    """Event-driven exact search implementation; selected config is stream-tested."""
    mask = ((cache['mags'] >= config['threshold_g']) & (cache['rise'] >= config['rise_g']) &
            (cache['snap'] >= config['snap_g']))
    points = np.flatnonzero(mask)
    ts, samples = cache['sample_times'], cache['samples']
    pending, last, output = None, -1e12, []
    emit_index = len(ts)

    def emit():
        nonlocal pending, last
        k = pending
        peak_index = int(cache['indices'][k])
        peak = samples[peak_index]
        available = samples[emit_index]
        last = peak['t']
        # rt.feature's only rejection is fewer than six available window rows.
        left = np.searchsorted(ts, peak['t']-80, side='left')
        right = min(np.searchsorted(ts, peak['t']+80, side='right'), emit_index+1)
        if right-left >= 6:
            output.append(dict(peak_ms=peak['t'], emitted_ms=available['t'], peak_g=peak['mag'],
                               peak_index=peak_index, available_index=int(emit_index)))
        pending = None

    for k in points:
        if pending is not None and int(cache['confirm'][k]) > emit_index:
            if emit_index < len(ts):
                emit()
        c = float(cache['times'][k])
        if c-last < config['refractory_ms']:
            continue
        if pending is None:
            pending = int(k)
            emit_index = max(int(cache['confirm'][k]), int(np.searchsorted(ts, c+config['context_ms'], side='left')))
        elif c-cache['times'][pending] < config['merge_ms'] and cache['mags'][k] > cache['mags'][pending]:
            pending = int(k)
            emit_index = max(int(cache['confirm'][k]), int(np.searchsorted(ts, c+config['context_ms'], side='left')))
        # A qualifying candidate may be confirmed on the emission packet.
        if pending is not None and int(cache['confirm'][k]) >= emit_index and emit_index < len(ts):
            emit()
    if pending is not None and emit_index < len(ts):
        emit()
    return output


def fast_detect(caches, config):
    output = []
    for hand in 'LR':
        output.extend(dict(hand=hand, **p) for p in fast_detect_one(caches[hand][config['confirm_samples']], config))
    output.sort(key=lambda p: (p['peak_ms'], p['hand']))
    for i, p in enumerate(output):
        p['prediction_id'] = i+1
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['raw', 'config', 'output']:
        ap.add_argument('--'+name, required=True)
    a = ap.parse_args()
    target = Path(a.output)
    assert not target.exists(), 'Refuse overwrite'
    config_file = json.loads(Path(a.config).read_text(encoding='utf-8'))
    assert config_file['runtime_contract'] == CONTRACT
    hands = rt.read_samples(a.raw)
    predictions, diagnostics = detect(hands, config_file['config'])
    target.parent.mkdir(parents=True, exist_ok=True)
    rt.write_csv(target, [{k: v for k, v in p.items() if k != 'feature'} for p in predictions])
    manifest = dict(runtime_contract=CONTRACT, config=config_file['config'],
                    raw_path=str(Path(a.raw).resolve()), raw_sha256=rt.sha256(a.raw),
                    config_sha256=rt.sha256(a.config), runtime_sha256=rt.sha256(__file__),
                    output_sha256=rt.sha256(target), rows=sum(map(len, hands.values())),
                    predictions=len(predictions), diagnostics=diagnostics,
                    midi_or_labels_read_during_detection=False, uses_fixed_recorded_time=True)
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
