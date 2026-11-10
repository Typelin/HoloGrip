"""Anchor rule 'consistent drum first, then relaxed snare': the existing rule picks the snare first (top-1 for all
first N hits), otherwise a drum the committee calls every time with mean >= 0.7. Variant: keep both, then, if
neither holds and the N hits are tight, anchor on the snare when it is within the top K for every hit."""
import json, sys
import numpy as np
import check_anchor2 as C2
SP = C2.SP; core, RD = SP.core, SP.RD
rt = core.runtime; E = rt.StreamEngine
ORIG_SETTLE, ORIG_REPLAY = E._settle_anchor, E._replay
K = {'k': 0}
def pick(self, P0, poses):
    med = np.median(poses, 0); tight = bool(np.all(np.hypot(*(poses-med).T) <= rt.ANCHOR_RULE['max_pose_deg']))
    if not tight: return None
    if all(self.anchor_drum in np.argsort(-p)[:1] for p in P0): return self.anchor_drum
    top = P0.argmax(1)
    if np.all(top == top[0]) and P0[:, top[0]].mean() >= rt.ANCHOR_RULE['any_drum_min_score']: return int(top[0])
    if K['k'] and all(self.anchor_drum in np.argsort(-p)[:K['k']] for p in P0): return self.anchor_drum
    return None
def settle(self, hand):
    st = self.stats[hand]; pend = self.pending[hand]; self.pending[hand] = []
    drum = pick(self, np.array([p0 for _, p0, _, _ in pend]), np.array([yp for _, _, _, yp in pend]))
    self.anchored[hand] = drum is not None; self.anchored_drum[hand] = drum
    for rot, p0, p1, _ in pend:
        if drum is not None: st[0].force(rot, drum); st[1].force(rot, drum)
        else: st[0].add(rot, p0, self.conf); st[1].add(rot, p1, self.conf)
E._settle_anchor = settle
# _replay (zero-direction rebuild) keeps the original rule; check how often it runs on these takes
def main():
    RD.setup(); _, config = core.load_components(); raws = SP.raw_paths()
    full = SP.load(SP.HERE/'out/model_ex.joblib'); lodo = {D: SP.load(RD.OUT/f'model_lodoex_{D}.joblib') for D in SP.FB.DAYS}
    res = {}
    for k in (0, 2, 3, 7):
        K['k'] = k
        F = SP.run({D: full for D in SP.FB.DAYS}, config, SP.BASE, raws); Lo = SP.run(lodo, config, SP.BASE, raws)
        core.runtime.attach_boosters(full); t = C2.today(full, config); res[k] = dict(full=F, lodo=Lo, today=t)
        print(f"snare_top{k}", f"full {F['ok']} fail {F['failed']} | lodo {Lo['ok']} fail {Lo['failed']} | today {t[0]}/{t[1]} fail {t[2]}", flush=True)
        for kk, r in (('full', F), ('lodo', Lo)):
            print('  ', kk, ' '.join(f"{x.replace('pretest:','')}={v['ok']}" for x, v in r['takes'].items()), flush=True)
        print('   today', {a: f'{v[0]}/{v[1]}' for a, v in t[3].items() if v[0] < v[1]}, flush=True)
    (RD.OUT/'score_anchor3.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
if __name__ == '__main__':
    main()
