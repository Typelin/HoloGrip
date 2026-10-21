"""Leave-one-DAY-out check (round 4 / 5 / 6), with and without pose features."""
import json, joblib, numpy as np, warnings
warnings.filterwarnings('ignore')
from hand_autolabel_v2 import fit, HIST
D=joblib.load('out/loro.joblib'); data=D['data']; names=D['names']
day={'fourth_song03':4,'fifth_song01':5,'fifth_song02':5,'fifth_song03':5,'sixth_S02':6,'sixth_S08':6}
cols={'all':np.arange(len(names)),'no_pose':np.array([i for i,n in enumerate(names) if '_rot' not in n])}
res={}
for mode,ci in cols.items():
    pred={}
    for d in (4,5,6):
        tr=[n for n in HIST if day[n]!=d]; te=[n for n in HIST if day[n]==d]
        Xtr=np.vstack([data[n]['X'][data[n]['y']>=0][:,ci] for n in tr]); ytr=np.concatenate([data[n]['y'][data[n]['y']>=0] for n in tr])
        ms=[fit(Xtr,ytr,s) for s in range(3)]
        for n in te: pred[n]=np.mean([m.predict_proba(data[n]['X'][:,ci])[:,1] for m in ms],0)
    out=[]
    for th in (.5,.9,.95,.98,.99):
        v=hv=ok=hok=cov=ev=0; per={}
        for n in HIST:
            d=data[n]; p=pred[n]; c=np.maximum(p,1-p)>=th; pr=(p>=.5).astype(int)
            vm=c&(d['y']>=0); hm=vm&d['hum']
            v+=vm.sum(); ok+=(pr[vm]==d['y'][vm]).sum(); hv+=hm.sum(); hok+=(pr[hm]==d['y'][hm]).sum(); cov+=c.sum(); ev+=len(p)
            per[n]=f"{(pr[hm]==d['y'][hm]).sum()}/{hm.sum()}"
        hum_total=sum(((data[n]['y']>=0)&data[n]['hum']).sum() for n in HIST)
        out.append(dict(th=th,acc=round(ok/v,4),hum_acc=round(hok/hv,4),hum_cov=round(hv/hum_total,4),coverage=round(cov/ev,4),hum_per=per))
        print(mode,out[-1],flush=True)
    res[mode]=out
    joblib.dump(pred,f'out/lodo_pred_{mode}.joblib')
json.dump(res,open('out/lodo.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
