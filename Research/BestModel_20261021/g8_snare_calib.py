"""Key-improvement test: explicit calibration taps (what commercial air-drum sticks do: a fixed reference
posture). Protocol: at the start of a take each hand taps the snare N times. Its median zero-relative
yaw/pitch is compared with the training snare centre of that hand; the difference corrects that hand's
drumhead zero for the rest of the take. Causal: only the first N hits are used; no MIDI, no future.
Evaluated on pretests whose first block per hand is snare (round5, round6, 01, 12, 15); round4 starts
with Hi-Hat so it is shown without calibration."""
import numpy as np, joblib
from scipy.spatial.transform import Rotation as Rot
import common as K, cells as Q
from phase3_aug import x_with_zero

STARTS_WITH_SNARE = ['pretest:round5', 'pretest:round6', 'seventh_01', 'seventh_12', 'seventh_15']


def yp(x):
    v = np.asarray(x[83:104]).reshape(7, 3)[4]*np.pi; e = Rot.from_rotvec(v).as_euler('ZYX', degrees=True); return e[0], e[1]


def main():
    D2 = joblib.load(K.OUT/'dataset2.joblib'); rows = D2['rows']
    centre = {h: np.median(np.array([yp(r['x']) for r in rows if r['hand'] == h and r['drum'] == '小鼓']), 0) for h in 'LR'}
    print('training snare centre', {h: np.round(c, 1) for h, c in centre.items()})
    com = K.train(rows, {}, seeds=(0, 1))
    for N in (0, 3, 5, 10):
        for use_pitch in (True, False):
            preds = {}
            for k, t in D2['tests'].items():
                P = []
                for h in 'LR':
                    it = sorted([c for c in t['items'] if c['hand'] == h], key=lambda c: c['t'])
                    dy = dp = 0.0
                    if N and k in STARTS_WITH_SNARE:
                        obs = np.median(np.array([yp(c['x']) for c in it[:N]]), 0); dy, dp = obs-centre[h]
                        if not use_pitch: dp = 0.0
                    X = np.stack([x_with_zero(c, dy, dp) for c in it]) if (dy or dp) else np.stack([c['x'] for c in it])
                    p = K.proba(com, h, X); P += [dict(t=c['t'], hand=h, drum=K.DRUMS[int(q.argmax())]) for c, q in zip(it, p)]
                preds[k] = P
            r = Q.score_cells(preds); print(f'N={N} pitch={use_pitch}', Q.fmt(r), flush=True)
            if N == 5 and use_pitch: joblib.dump(preds, K.OUT/'g8_preds_N5.joblib')


if __name__ == '__main__':
    main()
