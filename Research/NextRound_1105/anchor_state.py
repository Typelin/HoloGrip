import sys
import score_params as SP
core, RD = SP.core, SP.RD
RD.setup(); _, config = core.load_components(); raws = SP.raw_paths()
import csv
b = SP.load(SP.HERE/'out/model_ex.joblib')
for D in SP.FB.DAYS:
    for nm in SP.FB.PRETESTS[D]:
        tr = sorted(SP.L5.TRUTH[nm], key=lambda g: g['t'])
        first = {h: [g['drum'][:2] for g in tr if g['hand'] == h][:12] for h in 'LR'}
        hands = core.trajectory.read_raw(raws[nm]); src = list(csv.DictReader(open(raws[nm], encoding='utf-8-sig', newline='')))
        eng = core.Engine(b, config, snare_anchor=5)
        for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
            row = src[s['row_index']]; eng.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        rt = eng.runtime
        print(nm, {h: (rt.anchored[h], rt.classes[rt.anchored_drum[h]] if rt.anchored_drum[h] is not None else None) for h in 'LR'}, first, flush=True)
