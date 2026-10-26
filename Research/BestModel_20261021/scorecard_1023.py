"""Scorecard of the 8th-package engine on ALL round 4-7 recordings (sample-by-sample, exactly like the
field), scored like full_report (MIDI used, outputs, correct, wrong drum/hand, missed, extra), plus the
14 hand x drum cells over recordings with hand answers and over the pretests."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'第八次的包_暫定2026-10-27/Program'))
import seventh_core as core  # 8th package first
from verify_1023 import stream, TRUTH, match
import full_report as F
import common as K
import soft_hit_experiment as S
import numpy as np


def main():
    assert 'venue_free_runtime_1023' in core.runtime.__name__
    bundle, config = core.load_components(); rows = []; cells_rec = defaultdict(Counter); cells_pre = defaultdict(Counter)
    for name, title, role in F.ORDER:
        raw = S.C['tests'][name]['raw'] if name in K.TESTS else S.C['sessions'][name]['raw']
        P, _ = stream(bundle, config, raw)
        preds = [dict(t=p['peak_ms'], hand=p['hand'], drum=p['drum']) for p in P]
        used, excl = F.truth_of(name); res, cells = F.score(used, excl, preds)
        res.update(name=name, title=title, role=role); rows.append(res)
        for g, o in cells:
            if g['hand']: cells_rec[(g['hand'], g['drum'])][o] += 1
        if name in K.TESTS:
            truth = TRUTH[name]; pp = [dict(p) for p in P]; pairs = match(truth, pp)
            for i, g in enumerate(truth):
                j = pairs.get(i)
                o = 'missed' if j is None else ('wrong_hand' if g['hand'] and pp[j]['hand'] != g['hand'] else ('ok' if pp[j]['drum'] == g['drum'] else pp[j]['drum']))
                if g['hand']: cells_pre[(g['hand'], g['drum'])][o] += 1
        print(title, {k: res.get(k, 0) for k in ('midi', 'detected', 'correct', 'wrong_drum', 'wrong_hand', 'missed', 'extra', 'near_excluded')}, flush=True)
    def cell_rows(c):
        out = []
        for h in 'LR':
            for d in K.DRUMS:
                x = c[(h, d)]; n = sum(x.values()); mist = Counter({k: v for k, v in x.items() if k not in ('ok',)})
                out.append(dict(hand=h, drum=d, n=n, ok=x['ok'], acc=round(x['ok']/max(n, 1), 4), top=mist.most_common(2)))
        return out
    out = dict(recordings=rows, cells_all=cell_rows(cells_rec), cells_pretest=cell_rows(cells_pre))
    (K.OUT/'scorecard_1023.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=int), encoding='utf-8')
    for c in out['cells_pretest']: print('PRE', c)


if __name__ == '__main__':
    main()
