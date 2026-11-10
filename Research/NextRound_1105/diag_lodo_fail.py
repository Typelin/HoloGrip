import sys, json, numpy as np
sys.argv=['guard_eval.py','--sets','takes_lodo']
import guard_eval as G
from collections import Counter
J=[j for j in G.jobs() if j[1] in {'eighth_crash L','eighth_fioor ton L','eighth_ton2 L','seventh_08','sixth_S04','sixth_S06','seventh_14'}]
for kind,name,model,raw,truth in J:
    P,rt=G.stream(model,raw); ok,n,cells=G.VP.score(truth,P)
    pairs=G.VP.V.match(truth,P)
    conf=Counter((g['hand'],g['drum'],P[pairs[i]]['drum'] if i in pairs else 'miss') for i,g in enumerate(truth))
    print('##',name,ok,n,'zero_rot',rt.zero_rot,'rebuilds',rt.zero_rebuilds,'anch',rt.anchored)
    print('  ',[f"{h}:{t}->{p} x{c}" for (h,t,p),c in conf.most_common(8)])
    for h in 'LR':
        ph=[p for p in P if p['hand']==h][:8]
        print('   first',h,[(p['drum'][:2],round(p['model_score'],2)) for p in ph])
