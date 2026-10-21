"""Apply labeler v2 to today's songs; compare with conservative auto and GPT experimental candidates."""
import json, csv, joblib, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
from hand_autolabel_v2 import fit, features, HIST, EXP, TODAY
D=joblib.load('out/loro.joblib'); data=D['data']; names=D['names']
C=joblib.load(EXP/'實驗快取.joblib')
pose_free=np.array([i for i,n in enumerate(names) if '_rot' not in n]); allc=np.arange(len(names))
Xtr=np.vstack([data[n]['X'][data[n]['y']>=0] for n in HIST]); ytr=np.concatenate([data[n]['y'][data[n]['y']>=0] for n in HIST])
M={k:[fit(Xtr[:,c],ytr,s) for s in range(3)] for k,c in (('all',allc),('no_pose',pose_free))}
union=json.load(open(EXP/'配對實驗聯集_逐組.json',encoding='utf-8'))
exp={}
for g in union:
    for eid,h in zip(g['ids'],g['union_hands']): exp[(g['session'],str(eid))]=h
ix={n:i for i,n in enumerate(names)}
summary={}
for n in TODAY:
    s=C['sessions'][n]; ev,X,nm=features(s); assert nm==names
    pa=np.mean([m.predict_proba(X)[:,1] for m in M['all']],0); pn=np.mean([m.predict_proba(X[:,pose_free])[:,1] for m in M['no_pose']],0)
    conf=np.minimum(np.maximum(pa,1-pa),np.maximum(pn,1-pn)); agree=(pa>=.5)==(pn>=.5)
    hand=np.where(pa+pn>=1,'L','R')
    act=np.maximum(X[:,ix['L_max60']],X[:,ix['R_max60']]); ghost=(X[:,ix['vel_rel']]<.35)&(act<.8)
    cls=np.where(ghost,'疑似幽靈_建議刪除',np.where(agree&(conf>=.98),'自動',np.where(agree&(conf>=.90),'快速確認','需人工')))
    auto=s['auto']; rows=[]
    for k,e in enumerate(ev):
        a=auto.get(str(e['id']),{}).get('mark'); x=exp.get((n,str(e['id'])))
        rows.append(dict(id=e['id'],ordinal=e.get('ordinal'),midi_ms=round(e['t']),csv_ms=round(e['csv_t']),drum=e['name'],vel=e['vel'],
            suggested_hand=hand[k],confidence=round(float(conf[k]),4),models_agree=bool(agree[k]),category=cls[k],
            conservative_auto=a or '',experimental_candidate=x or '',L_act=round(float(X[k,ix['L_max60']]),2),R_act=round(float(X[k,ix['R_max60']]),2)))
    with open(f'out/{n}_自動手別建議_v2.csv','w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ca=[r for r in rows if r['conservative_auto']]; xa=[r for r in rows if r['experimental_candidate']]
    sm=dict(events=len(rows),**{c:int((cls==c).sum()) for c in ['自動','快速確認','需人工','疑似幽靈_建議刪除']},
        conservative=len(ca),conservative_agree=sum(r['suggested_hand']==r['conservative_auto'] for r in ca),
        conservative_in_auto=sum(r['category']=='自動' for r in ca),
        experimental=len(xa),experimental_agree=sum(r['suggested_hand']==r['experimental_candidate'] for r in xa),
        experimental_agree_auto=sum(r['suggested_hand']==r['experimental_candidate'] and r['category']=='自動' for r in xa),
        new_auto_beyond_conservative=sum(r['category']=='自動' and not r['conservative_auto'] for r in rows))
    summary[n]=sm; print(n,sm,flush=True)
json.dump(summary,open('out/today_summary.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
