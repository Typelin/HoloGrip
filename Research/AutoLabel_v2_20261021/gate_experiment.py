"""Inference-side gating: weak peaks (2.1-2.6 g) must be more confident to be output."""
import json, numpy as np
from pathlib import Path
import soft_hit_experiment as S, weight_experiment as W
lab=json.loads(S.LAB13.read_text(encoding='utf-8'))['labels']; r13,_=S.rows_from_labels('seventh_13',lab)
rows=[dict(hand=r['hand'],drum=r['drum'],x=np.asarray(r['x'],np.float32)) for r in S.C['baseline']]
rows+=[dict(hand=r['hand'],drum=r['drum'],x=np.asarray(r['x'],np.float32)) for r in S.C['auto'] if r['session'].startswith('seventh_') and r['session'] not in('seventh_13','seventh_14')]
rows+=[dict(hand=r['hand'],drum=r['drum'],x=r['x']) for r in r13]
for q in rows: q['g']=W.peak_g(q['x'])
ms=[W.train(rows,'W0',s) for s in range(2)]
T=S.tests(); res={}
def preds_g(items):
    out=[]
    for h in 'LR':
        it=[c for c in items if c['peak']['hand']==h]
        X=np.stack([np.asarray(c['x'],np.float32) for c in it]); cls=ms[0][h][0]; p=np.mean([S.G.proba(m[h][1],X) for m in ms],0)
        out+= [dict(t=c['peak']['peak_ms'],g=float(c['peak']['peak_g']),drum=cls[int(q.argmax())],score=float(q.max())) for c,q in zip(it,p)]
    return out
rules={'全部輸出':lambda p:True,'弱峰需≥0.8':lambda p:p['g']>=2.6 or p['score']>=.8,'全部需≥0.6':lambda p:p['score']>=.6,'強峰≥0.6、弱峰≥0.9':lambda p:p['score']>=(.6 if p['g']>=2.6 else .9)}
agg={r:dict(correct=0,total=0,out=0) for r in rules}
for k,(truth,items) in T.items():
    P=preds_g(items); res[k]={}
    for r,f in rules.items():
        keep=[dict(p,score=1.0) for p in P if f(p)]; sc=S.score(truth,keep,0.0)
        out=len(keep) if not k.startswith('song') else sc['answered']+0
        res[k][r]=dict(rate=sc['rate'],precision=sc['precision'],answered=sc['answered'],wrong=sc['wrong'])
        agg[r]['correct']+=sc['correct']; agg[r]['total']+=sc['total']; agg[r]['out']+=sc['answered']
    print(k,{r:f"{v['rate']:.3f}/{v['precision']:.3f}/{v['answered']}" for r,v in res[k].items()},flush=True)
for r,v in agg.items(): print('ALL',r,'命中率',round(v['correct']/v['total'],4),'輸出準確度',round(v['correct']/v['out'],4),'回答',v['out'],'/',v['total'])
Path('out/gate_experiment.json').write_text(json.dumps(dict(per=res,all=agg),ensure_ascii=False,indent=1),encoding='utf-8')
