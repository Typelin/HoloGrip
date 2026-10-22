"""Compare each take's TRUE per-drum centroid (yaw/pitch of the mid rotation block) with the training
templates of its leave-one-day-out model: is the difference a shift, a rotation or a per-drum change?"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import common as K
from seq_data import build

AB = ['SN', 'T1', 'T2', 'FT', 'HH', 'CR', 'RI']


def yp(R): return Rot.from_rotvec(R.reshape(-1, 7, 3)[:, 4]*np.pi).as_euler('ZYX', degrees=True)[:, :2]


seq, coms = build()
for n in ['pretest:round5', 'pretest:round6', 'seventh_12', 'seventh_15', 'pretest:round4', 'sixth_S02', 'seventh_13']:
    s = seq[n]; tm = coms[s['day']]['tm']
    for h, v in s['hands'].items():
        T = yp(tm[h]); Y = yp(v['x'][:, 83:104]); line = []
        for d in range(7):
            m = v['y'] == d
            if m.sum() < 3: continue
            c = Y[m].mean(0); line.append(f"{AB[d]} 樣板({T[d,0]:+5.0f},{T[d,1]:+4.0f}) 今天({c[0]:+5.0f},{c[1]:+4.0f}) 差({c[0]-T[d,0]:+4.0f},{c[1]-T[d,1]:+4.0f}) sd{Y[m].std(0).mean():3.0f}")
        print(n, h); print('   '+'\n   '.join(line))
