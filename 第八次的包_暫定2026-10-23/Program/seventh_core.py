"""Field adapter: validated packets -> external hits -> venue-free per-hit committee.

2026-10-23 (8th package): rounds 4-7 venue-free committee + causal per-drum offset
correction (earlier hits of the same hand only) + learned hit gate on 1.5 g candidates.
Each hit is reported as soon as the detector releases it (peak + 80 ms). No MIDI,
target sequence, opening taps, fitting or confidence rejection is used. The
caller saves every Raw packet before this quality validation and inference stage.
"""
import json
import math
from pathlib import Path

import joblib

import causal_hit_detector_1016 as detector
import absolute_world_motion_runtime_1019 as trajectory
import venue_free_runtime_1023 as runtime
from production_core import FrameValidator

PACK = Path(__file__).resolve().parents[1]
DRUMS = ['小鼓', '高音 Tom', '中音 Tom', '落地 Tom', 'Hi-Hat', 'Crash', 'Ride']
MODEL = PACK / 'Model/venue_free_committee_1023.joblib'
CONFIG = PACK / 'Model/detector_venue_free_1p5g.json'
MODEL_SHA256 = 'fd1d0d3f8f3fd55bfdf982578d897f901efdfc71880f95c88dabb8b4b361d1b1'
GENERATED_DIMENSIONS = runtime.INPUT_DIMENSIONS
SELECTED_DIMENSIONS = runtime.INPUT_DIMENSIONS
TRAINING_EVENTS = 13617
MODEL_NAME = '第4～7次場地無關模型＋打擊判斷＋逐鼓位置校正＋姿勢連續'


def load_guard():
    """Compatibility metadata only; no score/margin-based rejection."""
    return dict(enabled=False, min_score=0.0, min_margin=0.0,
                reason='venue_free_committee_without_abstention')


def load_components():
    manifest = json.loads((PACK / 'package_manifest.json').read_text(encoding='utf-8'))
    for path, entry in [(MODEL, manifest['model']), (CONFIG, manifest['detector'])]:
        if path.name != entry['file'] or detector.rt.sha256(path) != entry['sha256']:
            raise ValueError(f'檔案指紋不符，禁止開始：{path.name}')
    for relative, expected in manifest['runtime_hashes'].items():
        if detector.rt.sha256(PACK / relative) != expected:
            raise ValueError(f'程式指紋不符，禁止開始：{relative}')
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    if config['runtime_contract'] != detector.CONTRACT or config['detector_sha256'] != detector.rt.sha256(detector.__file__):
        raise ValueError('打擊偵測器契約或指紋不符')
    if detector.rt.sha256(MODEL) != MODEL_SHA256:
        raise ValueError('不是本次凍結的場地無關模型')
    bundle = joblib.load(MODEL)
    if bundle['contract'] != runtime.CONTRACT:
        raise ValueError('場地無關模型契約錯誤')
    if bundle['compass_heading_used'] or bundle['pretests_in_training']:
        raise ValueError('模型必須不用地磁方位、前測不進訓練')
    if set(bundle['drum_rotation_means']) != {f'{h}|{d}' for h in 'LR' for d in DRUMS}:
        raise ValueError('逐鼓位置校正資料不完整')
    if bundle['future_context_ms'] > 80 or list(bundle['pose_dts_ms']) != list(runtime.POSE_DTS):
        raise ValueError('模型時間窗契約錯誤')
    bundle['classes'] = [str(c) for c in bundle['classes']]
    if set(bundle['classes']) != set(DRUMS) or len(bundle['portable']) != 6:
        raise ValueError('模型類別或委員數錯誤')
    for member in bundle['portable'].values():
        for h in 'LR':
            if [str(c) for c in member[h]['classes']] != bundle['classes']:
                raise ValueError('委員類別順序不一致')
            if member[h]['mlp']['layers'][0]['W'].shape[0] != runtime.INPUT_DIMENSIONS:
                raise ValueError('模型特徵維度錯誤')
    if bundle['detector_config'] != config['config'] or manifest['output_guard']['enabled']:
        raise ValueError('打擊參數或輸出政策與研究版不一致')
    if 'hit_gate' not in bundle or bundle['hit_gate']['dims'] != 126:
        raise ValueError('打擊判斷模型缺失或維度錯誤')
    if bundle['training_events'] != TRAINING_EVENTS:
        raise ValueError('訓練筆數契約錯誤')
    runtime.attach_boosters(bundle)
    return bundle, config['config']


def packet_sample(packet, t, calibration, index):
    yaw, pitch = calibration
    return dict(t=t, ax=packet.ax, ay=packet.ay, az=packet.az,
                yaw=(packet.yaw-yaw+180) % 360-180, pitch=packet.pitch-pitch,
                roll=packet.roll, absolute_yaw=packet.yaw, absolute_pitch=packet.pitch,
                sensor_t=packet.sensor_time_ms,
                mag=math.sqrt(packet.ax**2+packet.ay**2+packet.az**2), index=index)


def raw_row(packet, s, index):
    return dict(sample_index=index, hand=packet.hand, received_time_ms=packet.received_time_ms,
                sensor_time_ms=packet.sensor_time_ms, packet_id=packet.packet_id,
                ax_g=packet.ax, ay_g=packet.ay, az_g=packet.az, accel_magnitude_g=s['mag'],
                yaw_deg=packet.yaw, pitch_deg=packet.pitch, roll_deg=packet.roll,
                cal_yaw_deg=s['yaw'], cal_pitch_deg=s['pitch'])


class Engine:
    def __init__(self, bundle, config):
        if config != bundle['detector_config']:
            raise ValueError('打擊參數必須等於凍結研究版')
        self.bundle = bundle
        self.config = dict(config)
        self.output_guard = load_guard()
        self.runtime = runtime.StreamEngine(bundle)
        self.streams = self.runtime.hit
        self.history = self.runtime.history
        self.validators = {h: FrameValidator() for h in 'LR'}
        self.invalid = {h: 0 for h in 'LR'}
        self.last_sensor_time = {h: None for h in 'LR'}
        self.count = 0

    def _reset_hand(self, hand):
        previous = self.streams[hand]
        fresh = detector.HitStream(hand, self.config)
        fresh.last_t, fresh.last_mag = previous.last_t, previous.last_mag
        self.streams[hand] = fresh
        self.history[hand].clear()
        self.last_sensor_time[hand] = None

    def push(self, hand, s, raw_frame):
        ok, reason = self.validators[hand].check(raw_frame)
        last = self.last_sensor_time[hand]
        if ok and (not math.isfinite(s['sensor_t']) or (last is not None and s['sensor_t'] <= last)):
            ok, reason = False, 'sensor_clock_reset_or_duplicate'
        if not ok:
            self.invalid[hand] += 1
            self._reset_hand(hand)
            return [], reason
        self.last_sensor_time[hand] = s['sensor_t']
        output = self.runtime.push(hand, s)
        self.count = self.runtime.count
        return [dict(**p, accepted=True, guard_reason='accepted_no_confidence_filter')
                for p in output], ''
