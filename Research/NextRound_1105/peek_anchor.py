import csv, sys
import numpy as np
import check_anchor3 as C3
SP = C3.SP; core, RD = SP.core, SP.RD; E = core.runtime.StreamEngine
LOG = []
orig = C3.pick
def pick(self, P0, poses):
    d = orig(self, P0, poses); med = np.median(poses, 0)
    LOG.append(dict(drum=d, spread=float(np.max(np.hypot(*(poses-med).T))),
                    top=[[self.classes[j][:2]+f'{p[j]:.2f}' for j in np.argsort(-p)[:3]] for p in P0], snare=[float(p[self.anchor_drum]) for p in P0]))
    return d
C3.pick = pick
RD.setup(); _, config = core.load_components(); raws = SP.raw_paths(); C3.K['k'] = 3
for nm, mp in (('seventh_15', 'out/model_lodoex_R7.joblib'), ('seventh_12', 'out/model_lodoex_R7.joblib')):
    b = SP.load(RD.OUT.parent/mp); LOG.clear()
    eng = core.Engine(b, config, snare_anchor=5); raw = raws[nm]; hands = core.trajectory.read_raw(raw); src = list(csv.DictReader(open(raw, encoding='utf-8-sig', newline='')))
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = src[s['row_index']]; eng.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
    for x in LOG: print(nm, x)
b = SP.load(RD.OUT.parent/'out/model_ex.joblib'); LOG.clear()
raw = str(C3.C2.TODAY); eng = core.Engine(b, config, snare_anchor=5); hands = core.trajectory.read_raw(raw); src = list(csv.DictReader(open(raw, encoding='utf-8-sig', newline='')))
for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
    row = src[s['row_index']]; eng.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
for x in LOG: print('today', x)
