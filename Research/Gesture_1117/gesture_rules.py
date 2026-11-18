"""Rule-based (no ML) gesture recogniser for one HoloGrip glove, causal, 100 Hz.

Per hand three gestures:
  1  揮      : thumb-web up, arm straight forward, swing outward (left hand: to the left) -> rotation about the
               vertical axis; the swing back is ignored (direction rule)
  2  敲（側）: thumb-web facing inward (left hand: right, right hand: left; palm down), wrist strikes down
  3  敲（上）: thumb-web up (palm sideways), wrist strikes down
  2 and 3 are the same motion; the hand pose just before the strike tells them apart.

Signals per sample
  R          sensor orientation from the glove's yaw/pitch/roll (intrinsic Z-Y-X, degrees; checked on 11/17 data:
             the gravity direction from R matches the accelerometer at rest within 1.4 deg)
  omega      world-frame angular velocity from consecutive orientations (deg/s); no Euler derivatives, so the
             near-vertical sensor pitch of the thumb-up pose (60..87 deg) causes no gimbal problems
  yaw_rate   omega about the vertical axis (+ = to the left for the glove's yaw sign), tilt_rate = |omega| horizontal
  acc        |a| in g
  gvec       gravity direction in the sensor frame from R (valid while moving, unlike the accelerometer)

Strike (2/3): decided at a local peak of acc >= strike_g (one sample after the peak). Over the strike_window_ms
  before the peak: d_tilt (sum of tilt_rate) >= strike_min_deg_<hand> and yaw share <= strike_share_max. Per hand
  because on 11/17 right-hand strikes turned >= 36 deg (standing, 15:23) while walking/gesturing noise peaks turned
  29..33 deg (right threshold 35);
  left-hand strikes go down to 25 deg, so the left threshold stays low. A strong peak (>= strike_strong_g, 4.5 g)
  may have yaw share up to strike_share_max_strong (0.7): standing strikes at 15:10/15:23 on 11/17 turned about as
  much sideways as down (share ~0.5) but peaked at 4.5..8 g, while acc peaks inside swings stayed below 4.5 g
  except 3 of ~1600 (that also passed the other strike tests). A strong peak also needs only strike_min_deg_strong
  (30) of downward turn: standing right-hand ③ strikes at 15:23 peaked at 4.5..7.6 g but turned 28..35 deg, while
  the noise peaks that set the right-hand 35 deg stayed below 3.4 g (all 11/17 recordings: +10 correct, -1 wrong).
  Tried and not used (11/17 data): a longer swing hold (120..250 ms: real swings end with a 4..6 g stop peak
  130..280 ms after the decision, so they got vetoed), a dead zone around the ②/③ split that outputs nothing
  (a dropped strike sets no locks, so its rebound was read as another gesture: more wrong outputs).
  Pose = gvec strike_pose_ms before the peak, read as the rotation about the forearm: the axis is the one that
  turns the calibrated 'side' pose into the 'up' pose (cross product), and the rotation is the angle of gvec
  around it, 0 at 'side' and growing towards 'up'. ③ when the rotation >= the split, else ②. The forearm's
  elevation (along the axis) no longer counts, unlike the nearest-pose angle used before.
  The split adapts: it starts at pose_split_init x the calibrated side->up rotation and then sits halfway between
  the medians of the last pose_adapt_n ② and ③ rotations, kept within [pose_split_min, pose_split_max] x that
  rotation, once each side has pose_adapt_min (2) clear calls. Only clear calls (at least pose_adapt_margin x the rotation away from the split) update the medians,
  so a few wrong calls cannot drag the split towards themselves. On the 11/17 recordings ② and ③ were always ~60 deg apart, but the whole left-hand range shifted by
  ~35 deg between 15:04 and 15:10 (② median -37, ③ 24 deg vs -2 / 62 earlier) with the same calibration, so a fixed
  split failed there. Pose-only result on the detected strikes of all 7 worn 11/17 recordings: nearest pose
  879/1027 (85.6%), adaptive rotation split 1011/1027 (98.4%).
Swing (1): a movement opens when |omega| > trig_dps; while it lasts, the outward yaw is summed one-sidedly (the sum
  restarts whenever the hand turns back inward) together with the tilt over the same run; as soon as the run reaches
  swing_min_deg with yaw share >= swing_share, the swing is reported (early decision). The movement ends after |omega| < release_dps for release_ms or max_ms.
Swing confirm: a detected swing is held for swing_confirm_ms (80) before it is output. If a strike peak
  >= swing_veto_g (5 g) passes the strike tests in that time, the swing is dropped and the strike is output
  instead. For such a peak the yaw-share test and the strike locks (refractory, swing->strike) are skipped and the
  downward turn only needs swing_veto_min_deg (15): fast ③ strikes in a row otherwise cascade (one ③ read as a
  swing locks the next strike out for 800 ms, whose own outward turn is read as the next swing), and those
  right-hand ③ strikes turned only 16..34 deg down (walking/gesturing noise peaks stayed below 3.4 g). 11/17: right-hand ③ strikes in the standing recordings turned outward
  enough to pass as a swing (25 times) and 19 of them had a >= 5 g peak within 80 ms of the swing decision; real
  swings had one 2 times out of 325. Cost: swings come out 80 ms later.
Glitch guard: a sample implying |omega| > glitch_dps (2000 deg/s) is skipped (Euler jump near pitch 90 deg).
Debounce (checked on the 11/17 recordings; strikes in a row come as fast as 210 ms apart, swings >= 530 ms):
  strike -> strike   refractory_ms (250): at most 4 strikes per second, so fast shaking does not flood the output
  class switch       within a burst (previous candidate < switch_gap_ms (600) ago) a different gesture is output
                     only after it shows up switch_confirm (2) times in a row; a single odd one is dropped, so a
                     burst never alternates 2-3-2-3, and a wrong first call still recovers on the next strikes
  swing  -> strike   swing_to_strike_ms (800): ① and ③ share the thumb-web-up pose, and the follow-through / return
                     of a right-hand swing read as a ③ strike 280..1300 ms later; those strikes turn 46..50 deg down,
                     which overlaps real right-hand strikes (>= 46 deg), so a time lock is used instead of a threshold
  strike -> swing    strike_to_swing_ms (500): the wrist's rebound / sideways drift between fast strikes once read
                     as a swing
  swing  -> swing    swing_refractory_ms (450) AND the hand must first turn back inward by swing_rearm_deg (20):
                     a single long swing with a small wobble once fired twice 200 ms apart
  swing pose         a swing starting more than swing_up_max_deg (80) from the calibrated thumb-web-up pose is
                     not a swing (11/17: real swings <= 72 deg; sideways flicks in ② segments were 83..137 deg)
"""
from collections import deque
import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot

GESTURES = {1: '揮', 2: '敲（虎口朝側）', 3: '敲（虎口朝上）'}
SIDE_WORD = {'L': '朝右', 'R': '朝左'}
DEFAULT_PARAMS = dict(trig_dps=150.0, release_dps=80.0, release_ms=40.0, max_ms=800.0, refractory_ms=250.0, switch_gap_ms=600.0, switch_confirm=2.0, swing_refractory_ms=450.0, strike_to_swing_ms=500.0, swing_rearm_deg=20.0, swing_up_max_deg=80.0, swing_to_strike_ms=800.0,
                      swing_min_deg=40.0, swing_share=0.6, swing_dir_L='+', swing_dir_R='-',
                      strike_g=2.2, strike_min_deg_L=25.0, strike_min_deg_R=35.0, strike_share_max=0.4, strike_strong_g=4.5, strike_share_max_strong=0.7, strike_window_ms=200.0,
                      strike_pose_ms=150.0, pose_max_deg=90.0, glitch_dps=2000.0,
                      pose_split_init=0.4, pose_split_min=0.15, pose_split_max=0.65, pose_adapt_n=6.0, pose_adapt_min=2.0, pose_adapt_margin=0.1,
                      swing_confirm_ms=80.0, swing_veto_g=5.0, swing_veto_min_deg=15.0, strike_min_deg_strong=30.0, pose_dead_zone=0.0, swing_veto_g_low=0.0, swing_veto_tilt=15.0)


def gesture_name(hand, g):
    if g == 2: return f'敲（虎口{SIDE_WORD[hand]}）'
    return GESTURES.get(g, '—')


def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v/n if n > 1e-9 else v


def angle_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(unit(a), unit(b)))))))


def gravity_of(yaw, pitch, roll):
    return Rot.from_euler('ZYX', [yaw, pitch, roll], degrees=True).inv().apply([0.0, 0.0, 1.0])


class HandRules:
    def __init__(self, hand, params=None, poses=None):
        self.hand = hand; self.p = dict(DEFAULT_PARAMS, **(params or {}))
        self.poses = dict(poses or {})          # 'up' / 'side' -> unit gravity vector in the sensor frame
        self.reset()

    def reset(self):
        self.prev_R = None; self.prev_t = None
        self.hist = deque(maxlen=120)           # (t, dt, yaw_rate, tilt_rate, acc, gvec)
        self.mv = None; self.glitches = 0
        self.block_strike = self.block_swing = -1e18; self.armed = True; self.back = 0.0
        self.cur_g = 0; self.last_cand = -1e18; self.pend_g = 0; self.pend_n = 0
        self.pend_swing = None
        self.rot2 = deque(maxlen=int(self.p['pose_adapt_n'])); self.rot3 = deque(maxlen=int(self.p['pose_adapt_n']))

    def set_params(self, params): self.p = dict(DEFAULT_PARAMS, **params)

    def strike_min_deg(self): return self.p[f'strike_min_deg_{self.hand}']

    def pose_of(self, g):
        if 'up' not in self.poses or 'side' not in self.poses or g is None: return '?', None, None
        au, as_ = angle_deg(g, self.poses['up']), angle_deg(g, self.poses['side'])
        if min(au, as_) > self.p['pose_max_deg']: return '?', au, as_
        return ('up' if au <= as_ else 'side'), au, as_

    def rotation_of(self, g):
        """(rotation about the forearm in deg: 0 = 'side', + towards 'up'; calibrated side->up rotation) or None."""
        if 'up' not in self.poses or 'side' not in self.poses or g is None: return None
        side, up = unit(self.poses['side']), unit(self.poses['up'])
        ax = unit(np.cross(side, up)); u = unit(side-ax*np.dot(side, ax)); v = np.cross(ax, u)
        def rot(x):
            x = np.asarray(x, float); xp = x-ax*np.dot(x, ax)
            return math.degrees(math.atan2(float(np.dot(xp, v)), float(np.dot(xp, u))))
        return rot(g), rot(up)

    def pose_split(self, full):
        p = self.p; s = p['pose_split_init']*full
        if min(len(self.rot2), len(self.rot3)) >= p['pose_adapt_min']:
            s = (float(np.median(self.rot2))+float(np.median(self.rot3)))/2
        return min(p['pose_split_max']*full, max(p['pose_split_min']*full, s))

    def push(self, t_ms, ax, ay, az, yaw, pitch, roll):
        """Feed one sample (t_ms on the glove clock). Returns (decided event dict or None, live signals)."""
        if self.prev_t is not None and (t_ms < self.prev_t or t_ms-self.prev_t > 1000):
            self.reset()        # glove clock went back or a long gap (glove restarted, seen 11/17 right hand)
        acc = float(math.sqrt(ax*ax+ay*ay+az*az))
        R = Rot.from_euler('ZYX', [yaw, pitch, roll], degrees=True); gvec = R.inv().apply([0.0, 0.0, 1.0])
        sig = dict(t=t_ms, speed=0.0, yaw_rate=0.0, tilt_rate=0.0, acc=acc)
        out = None
        if self.prev_R is not None:
            dt = (t_ms-self.prev_t)/1000.0
            if 0.002 <= dt <= 0.06:
                w = np.degrees((R*self.prev_R.inv()).as_rotvec())/dt
                yr = float(w[2]); tr = float(math.hypot(w[0], w[1])); sp = float(math.sqrt(w @ w))
                if sp > self.p['glitch_dps'] and self.glitches < 3:
                    # the glove's Euler output jumps near pitch 90 deg (thumb-web-up pose, seen 11/17): skip the
                    # sample and compare the next one with the last good orientation
                    self.glitches += 1
                    return None, sig
                self.glitches = 0
                sig.update(speed=sp, yaw_rate=yr, tilt_rate=tr)
                self.hist.append((t_ms, dt, yr, tr, acc, gvec))
                # both detectors see every sample (a rejected strike peak must not cut the swing's integration);
                # an accepted gesture wins over a rejected one
                es = self._strike(t_ms); ew = self._swing(t_ms, dt, sp, yr, tr, acc)
                if ew is None and self.pend_swing is not None and t_ms >= self.pend_swing[0]:
                    e, self.pend_swing = self.pend_swing[1], None; ew = self._finish(e, 1, '', t_ms)
                out = next((e for e in (es, ew) if e and e['gesture']), es or ew)
            else:
                self.hist.clear(); self.mv = None
        self.prev_R, self.prev_t = R, t_ms
        return out, sig

    # ---------------- strike: local acc peak
    def _strike(self, t):
        p = self.p; h = self.hist
        if len(h) < 3: return None
        a2, a1, a0 = h[-3][4], h[-2][4], h[-1][4]
        if not (a1 >= p['strike_g'] and a1 >= a2 and a1 > a0): return None
        tp = h[-2][0]
        veto = self.pend_swing is not None and (a1 >= p['swing_veto_g'] or (
            p['swing_veto_g_low'] > 0 and a1 >= p['swing_veto_g_low'] and self.pend_swing[1]['d_tilt'] >= p['swing_veto_tilt']))
        if tp < self.block_strike and not veto: return None
        win = [x for x in h if tp-p['strike_window_ms'] <= x[0] <= tp]
        d_yaw = sum(x[2]*x[1] for x in win); d_tilt = sum(x[3]*x[1] for x in win)
        share = abs(d_yaw)/(abs(d_yaw)+d_tilt+1e-9)
        if self.pend_swing is not None and not veto: return None     # a held swing blocks ordinary strikes as before
        before = [x for x in h if x[0] <= tp-p['strike_pose_ms']]
        gvec = before[-1][5] if before else h[0][5]
        pose, au, as_ = self.pose_of(gvec)
        rf = self.rotation_of(gvec); rot = split = None
        if rf is not None:
            rot, full = rf; split = self.pose_split(full); pose = 'up' if rot >= split else 'side'
            if abs(rot-split) < p['pose_dead_zone']*full: pose = 'unsure'
        e = dict(hand=self.hand, kind='strike', t0=tp-p['strike_window_ms'], t1=t, dur_ms=t-(tp-p['strike_window_ms']),
                 peak_t=tp, d_yaw=d_yaw, d_tilt=d_tilt, yaw_share=share, peak_g=a1,
                 peak_dps=max(math.hypot(x[2], x[3]) for x in win) if win else 0.0, pose=pose, ang_up=au, ang_side=as_,
                 pre_gvec=[float(v) for v in gvec], rotation=rot, split=split)
        g = 0; why = ''
        min_deg = p['swing_veto_min_deg'] if veto else self.strike_min_deg()
        if a1 >= p['strike_strong_g'] and p['strike_min_deg_strong'] > 0: min_deg = min(min_deg, p['strike_min_deg_strong'])
        if d_tilt < min_deg: why = f'敲的幅度不足（往下 {d_tilt:.0f}°）'
        elif not veto and share > (p['strike_share_max_strong'] if a1 >= p['strike_strong_g'] else p['strike_share_max']): why = f'有尖峰但主要是水平轉（占比 {share:.2f}）'
        elif pose == 'up':
            g = 3
            if rot is not None and rot-split >= self.p['pose_adapt_margin']*full: self.rot3.append(rot)
        elif pose == 'side':
            g = 2
            if rot is not None and split-rot >= self.p['pose_adapt_margin']*full: self.rot2.append(rot)
        elif pose == 'unsure': why = f'②/③ 不確定（旋轉 {rot:.0f}°，門檻 {split:.0f}°），寧可不輸出'
        else: why = '敲但姿勢不明（先做姿勢校正）'
        if g and self.pend_swing is not None:
            self.pend_swing = None; e['reason_extra'] = '揮確認期間出現強敲，改判為敲'
        return self._finish(e, g, why, t)

    # ---------------- swing: movement with early decision
    def _swing(self, t, dt, sp, yr, tr, acc):
        p = self.p
        sign = p[f'swing_dir_{self.hand}']; oy = yr if sign == '+' else -yr if sign == '-' else abs(yr)
        if not self.armed:      # re-arm once the hand has turned back inward after the last swing
            self.back = max(0.0, self.back-oy*dt)
            if self.back >= p['swing_rearm_deg'] or sign == 'any': self.armed = True
        if self.mv is None:
            if sp <= p['trig_dps']: return None
            self.mv = dict(t0=t, d_yaw=0.0, d_tilt=0.0, peak_g=acc, peak_dps=sp, quiet=None, done=False,
                           gvec=self.hist[0][5] if self.hist else None)
        m = self.mv
        m['d_yaw'] += yr*dt; m['d_tilt'] += tr*dt; m['peak_g'] = max(m['peak_g'], acc); m['peak_dps'] = max(m['peak_dps'], sp)
        # outward run: one-sided sum of the outward yaw (restarts whenever the hand turns back inward), so a swing
        # that follows the swing back without a pause (continuous back-and-forth, seen 11/17 right hand) still counts
        if m.setdefault('run', 0.0) + oy*dt <= 0: m['run'] = 0.0; m['run_tilt'] = 0.0
        else: m['run'] += oy*dt; m['run_tilt'] = m.get('run_tilt', 0.0)+tr*dt
        m['quiet'] = (m['quiet'] or t) if sp < p['release_dps'] else None
        ended = (m['quiet'] is not None and t-m['quiet'] >= p['release_ms']) or t-m['t0'] >= p['max_ms']
        out = None
        if not m['done']:
            outward = m['run']; dy = outward if sign != '-' else -outward
            share = outward/(outward+m['run_tilt']+1e-9)
            if outward >= p['swing_min_deg'] and share >= p['swing_share'] and t >= self.block_swing and self.armed:
                m['done'] = True
                pose, au, as_ = self.pose_of(m['gvec'])
                if au is not None and au > p['swing_up_max_deg']:
                    if ended: self.mv = None
                    return None
                e = dict(hand=self.hand, kind='swing', t0=m['t0'], t1=t, dur_ms=t-m['t0'], peak_t=t, d_yaw=dy,
                         d_tilt=m['run_tilt'], yaw_share=share, peak_g=m['peak_g'], peak_dps=m['peak_dps'],
                         pose=pose, ang_up=au, ang_side=as_)
                if p['swing_confirm_ms'] > 0: self.pend_swing = (t+p['swing_confirm_ms'], e)
                else: out = self._finish(e, 1, '', t)
        if ended: self.mv = None
        return out

    def _gate(self, g, t):
        """Class hysteresis inside a burst. Returns (gesture to output or 0, reason)."""
        p = self.p; in_burst = t-self.last_cand < p['switch_gap_ms']; self.last_cand = t
        if in_burst and self.cur_g and g != self.cur_g:
            if self.pend_g == g: self.pend_n += 1
            else: self.pend_g, self.pend_n = g, 1
            if self.pend_n < p['switch_confirm']:
                return 0, f'連發中換成{gesture_name(self.hand, g)}，再出現一次才換（維持{gesture_name(self.hand, self.cur_g)}）'
        self.cur_g = g; self.pend_g = 0; self.pend_n = 0
        return g, ''

    def _finish(self, e, g, why, t):
        raw = g
        if g: g, why = self._gate(g, t)
        e['gesture'] = g; e['reason'] = why; e['name'] = gesture_name(self.hand, g) if g else '不算'
        e['raw_gesture'] = raw
        if raw:     # a dropped switch still debounces like an accepted gesture (the movement happened)
            g = raw; p = self.p
            self.block_strike = max(self.block_strike, t+p['refractory_ms'])
            if g == 1:
                self.block_swing = t+p['swing_refractory_ms']; self.armed = False; self.back = 0.0
                self.block_strike = max(self.block_strike, t+p['swing_to_strike_ms'])
            else:
                self.block_swing = max(self.block_swing, t+p['strike_to_swing_ms'])
            if self.mv is not None: self.mv['done'] = True     # one gesture per movement
        return e


def run_file(rows, params, poses):
    """Offline: rows = dicts with hand, sensor_time_ms, ax_g.. roll_deg, label. Returns decided events with label."""
    eng = {h: HandRules(h, params, poses.get(h, {})) for h in 'LR'}; out = []
    for r in rows:
        h = r['hand']
        if h not in eng: continue
        e, _ = eng[h].push(float(r['sensor_time_ms']), float(r['ax_g']), float(r['ay_g']), float(r['az_g']),
                           float(r['yaw_deg']), float(r['pitch_deg']), float(r['roll_deg']))
        if e: e['label'] = r.get('label', ''); out.append(e)
    return out
