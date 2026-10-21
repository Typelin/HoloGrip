"""Total table: new rounds4-7 model vs round-6 (deployed 10/20, rounds 4-6) model.
Per recording: accuracy on labelled hits (hand known). Pretests: full pipeline vs MIDI.
Ablation: identical settings trained on rounds 4-6 only vs rounds 4-7 (2 seeds)."""
import json
from collections import defaultdict
import numpy as np, joblib
import common as K
import venue_free_runtime_1020 as rt
from final_model import runtime_exam

NAMES = {'fourth_song03': '第4次 四鼓強化', 'fifth_song01': '第5次 歌曲1', 'fifth_song02': '第5次 歌曲2', 'fifth_song03': '第5次 四鼓強化',
         'sixth_S02': '第6次 歌曲2', 'sixth_S08': '第6次 四鼓強化', 'sixth_S04': '第6次 單手 左Hi-Hat', 'sixth_S05': '第6次 單手 左Crash',
         'sixth_S06': '第6次 單手 左Ride', 'sixth_S07': '第6次 單手 右Crash', 'seventh_03': '第7次 歌曲1 原鼓位', 'seventh_04': '第7次 歌曲2 原鼓位',
         'seventh_05': '第7次 四鼓強化', 'seventh_06': '第7次 單手 左高音Tom', 'seventh_07': '第7次 單手 左中音Tom', 'seventh_08': '第7次 單手 左落地Tom',
         'seventh_09': '第7次 單手 右高音Tom', 'seventh_10': '第7次 單手 左Crash', 'seventh_11': '第7次 單手 左小鼓', 'seventh_13': '第7次 歌曲1 移動後', 'seventh_14': '第7次 歌曲2 移動後'}


def rec_acc(bundle, rows):
    rt.attach_boosters(bundle); out = {}
    by = defaultdict(list)
    for r in rows: by[r['session']].append(r)
    for s, rs in by.items():
        ok = 0
        for h in 'LR':
            t = [r for r in rs if r['hand'] == h]
            if not t: continue
            p = rt.probabilities(bundle, np.stack([r['x'] for r in t]), h)
            ok += sum(bundle['classes'][int(q.argmax())] == r['drum'] for q, r in zip(p, t))
        out[s] = (ok, len(rs))
    return out


def main():
    D = K.build(); rows = D['rows']
    new = joblib.load(K.OUT/'venue_free_committee_rounds4to7_20261021.joblib'); old = joblib.load(K.P7/'Model/venue_free_committee_1020.joblib')
    for b in (new, old): b['classes'] = [str(c) for c in b['classes']]
    rn, ro = rec_acc(new, rows), rec_acc(old, rows)
    pn, po = runtime_exam(new, D), runtime_exam(old, D)
    # ablation with identical settings
    abl = {}
    for name, tr in [('R4-6', [r for r in rows if r['group'] in ('R4', 'R5', 'R6')]), ('R4-7', rows)]:
        com = K.train(tr, {}, seeds=(0, 1)); res, summ = K.exam(com, D, thrs=(0.0, 0.6))
        abl[name] = dict(per={k: round(v['0.0']['correct']/v['0.0']['total'], 4) for k, v in res.items()}, summ=summ)
        print('ablation', name, abl[name], flush=True)
    out = dict(recordings={s: dict(name=NAMES[s], new=rn[s], old=ro[s]) for s in NAMES},
               pretests={k: dict(new=[pn[k][0.0]['correct'], pn[k][0.0]['total']], old=[po[k][0.0]['correct'], po[k][0.0]['total']]) for k in pn}, ablation=abl)
    (K.OUT/'total_table.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=int), encoding='utf-8')
    for s in NAMES: print(NAMES[s], rn[s], ro[s])
    for k in pn: print(k, out['pretests'][k])


if __name__ == '__main__':
    main()
