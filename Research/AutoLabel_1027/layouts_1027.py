"""Per-day drum layouts (rotation-block means of re-centred training rows, per hand and drum) for each
cached setting; drums a day lacks fall back to the pooled mean. Saved next to the model cache."""
import sys
from pathlib import Path
import numpy as np, joblib
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mix_1027 as MX
RC, FB, L5, K = MX.RC, MX.FB, MX.L5, MX.K
ROT = RC.ROT

def main():
    rows, *_ = FB.setup()
    for tag in ['full']+FB.DAYS:
        hold = set() if tag == 'full' else {tag}
        tr = RC.recenter([r for r in rows if L5.day_of(r['session']) not in hold]); pooled = MX.means(tr); lay = {}
        for D in [d for d in FB.DAYS if d not in hold]:
            lay[D] = {}
            for h in 'LR':
                m = pooled[h].copy(); have = np.zeros(7, bool)
                for i, d in enumerate(K.DRUMS):
                    v = [r['x'][ROT] for r in tr if L5.day_of(r['session']) == D and r['hand'] == h and r['drum'] == d]
                    if len(v) >= 10: m[i] = np.mean(v, 0); have[i] = True
                lay[D][h] = dict(mean=m, have=have)
        joblib.dump(lay, HERE/f'out/cache/layouts_{tag}.joblib'); print('layouts', tag, list(lay), flush=True)

if __name__ == '__main__':
    main()
