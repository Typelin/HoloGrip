"""Live drum judgement for the raw collector (display + side file only).

The collector keeps writing every raw packet exactly as before; this module never touches the raw CSV,
the recorder queue or the packets. Packets are copied into a bounded queue and a worker thread runs the
same field engine as the pretest program (seventh_core.Engine with the automatic snare anchor, causal,
fingerprint-checked package model). Each take gets a fresh engine and writes
即時判斷/<raw stem>_即時判斷.csv in a subfolder of the raw folder (so raw-file globs such as
'Raw/*_raw_100hz_*.csv' never pick it up). If the model cannot be loaded or the queue overflows,
only the judgement is affected (and reported in the manifest); recording continues.
"""
from __future__ import annotations

import csv
import queue
import threading
import time
from collections import Counter, deque
from pathlib import Path

from threadpoolctl import threadpool_limits

import seventh_core as core

QUEUE_MAX = 30000          # about 150 s of both hands at 100 Hz
RECENT = 8                 # last decisions shown per hand
FIELDS = ['prediction_id', 'hand', 'peak_ms', 'emitted_ms', 'peak_g', 'drum', 'model_score', 'model_margin',
          'accepted', 'guard_reason', 'hit_probability', 'processing_latency_ms']


class LiveJudge:
    def __init__(self, log=lambda message: None, clock=time.monotonic):
        self.log = log; self.clock = clock
        self.bundle = self.config = None; self.load_error = None; self.ready = threading.Event()
        self.q: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self.lock = threading.Lock()
        self.take = None
        threading.Thread(target=self._load, daemon=True, name='judge-load').start()
        threading.Thread(target=self._work, daemon=True, name='judge-work').start()

    # ---- model -------------------------------------------------------------------------------
    def _load(self):
        try:
            with threadpool_limits(limits=1):
                self.bundle, self.config = core.load_components()
            self.log(f'即時判斷模型已載入（{core.MODEL_SHA256[:12]}）；錄製時會即時顯示每一下判成哪顆鼓')
        except Exception as exc:
            self.load_error = str(exc)[:200]
            self.log(f'⚠ 即時判斷模型無法載入：{self.load_error}；仍可正常收集 Raw，只是不顯示判斷')
        finally:
            self.ready.set()

    @property
    def available(self):
        return self.bundle is not None

    # ---- take lifecycle ----------------------------------------------------------------------
    def start(self, t0, calibration, raw_path):
        """New take: t0 = recorder start (monotonic), calibration = {hand: (yaw, pitch)}."""
        self.stop()
        if not self.ready.is_set() or not self.available:   # never block the take start on model loading
            with self.lock:
                self.take = None
            return False
        path = None
        if raw_path:
            raw = Path(raw_path); path = raw.parent/'即時判斷'/f'{raw.stem}_即時判斷.csv'; path.parent.mkdir(exist_ok=True)
        f = path.open('w', encoding='utf-8-sig', newline='') if path else None
        w = csv.DictWriter(f, fieldnames=FIELDS) if f else None
        if w: w.writeheader()
        take = dict(t0=t0, cal={h: tuple(calibration[h]) for h in 'LR'}, index={h: 0 for h in 'LR'},
                    engine=core.Engine(self.bundle, self.config, snare_anchor=core.runtime.PRETEST_SNARE_ANCHOR),
                    file=f, writer=w, path=path, dropped=0, errors=0, error=None, n=0,
                    counts={h: Counter() for h in 'LR'}, recent={h: deque(maxlen=RECENT) for h in 'LR'},
                    last={h: None for h in 'LR'}, latency=deque(maxlen=500), active=True)
        with self.lock:
            self.take = take
        return True

    def feed(self, packet):
        take = self.take
        if take is None or not take['active']: return
        try:
            self.q.put_nowait((take, packet))
        except queue.Full:
            take['dropped'] += 1

    def stop(self, timeout=5.0):
        """End the take: judge what is still queued (up to timeout), close the file, return a summary."""
        with self.lock:
            take = self.take
        if take is None: return None
        take['active'] = False
        end = self.clock()+timeout
        while not self.q.empty() and self.clock() < end: time.sleep(0.02)
        with self.lock:
            if take['file']: take['file'].close(); take['file'] = None
            self.take = None
        lat = sorted(take['latency'])
        return dict(predictions_csv=str(take['path']) if take['path'] else None, predictions=take['n'],
                    by_hand={h: dict(take['counts'][h]) for h in 'LR'}, queue_dropped=take['dropped'],
                    unjudged_at_stop=self.q.qsize(), errors=take['errors'], first_error=take['error'],
                    latency_ms_median=lat[len(lat)//2] if lat else None, model_sha256=core.MODEL_SHA256,
                    snare_anchor=dict(take['engine'].runtime.anchored), display_only=True)

    # ---- worker ------------------------------------------------------------------------------
    def _work(self):
        with threadpool_limits(limits=1):
            while True:
                take, packet = self.q.get()
                with self.lock:
                    if take is not self.take or take['file'] is None and take['writer'] is not None: continue
                    try: self._push(take, packet)
                    except Exception as exc:
                        take['errors'] += 1
                        if take['error'] is None: take['error'] = f'{type(exc).__name__}: {exc}'[:200]

    def _push(self, take, packet):
        if packet.received_monotonic < take['t0']: return
        h = packet.hand
        t = max(0, int(round((packet.received_monotonic-take['t0'])*1000)))
        sample = core.packet_sample(packet, t, take['cal'][h], take['index'][h]); take['index'][h] += 1
        frame = dict(ax=packet.ax, ay=packet.ay, az=packet.az, yaw=packet.yaw, pitch=packet.pitch, roll=packet.roll)
        predictions, _ = take['engine'].push(h, sample, frame)
        for p in predictions:
            p['processing_latency_ms'] = round(max(0., self.clock()-packet.received_monotonic)*1000, 3)
            take['latency'].append(p['processing_latency_ms'])
            if take['writer']: take['writer'].writerow({k: p.get(k) for k in FIELDS})
            if p.get('accepted', True):
                take['n'] += 1; take['counts'][h][p['drum']] += 1
                take['recent'][h].append(p['drum']); take['last'][h] = p
        if predictions and take['file']: take['file'].flush()

    # ---- display -----------------------------------------------------------------------------
    def view(self, hand):
        """Text for the hand card while recording, or None when nothing to show."""
        if not self.ready.is_set(): return '即時判斷：模型載入中…'
        if not self.available: return '即時判斷：未啟用（模型無法載入）'
        take = self.take
        if take is None: return None
        last = take['last'][hand]
        if last is None: return '即時判斷：等待第一下…'
        counts = take['counts'][hand]
        top = ' · '.join(f'{d} {n}' for d, n in counts.most_common(4))
        return (f"即時判斷：{last['drum']}（{last['model_score']:.2f}）\n"
                f"最近：{' '.join(d.replace(' Tom', '') for d in take['recent'][hand])}\n"
                f"本段 {sum(counts.values())} 下：{top}")
