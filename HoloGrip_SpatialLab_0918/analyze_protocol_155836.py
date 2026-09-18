from pathlib import Path
import csv, math, json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import numpy as np

S=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918\protocol_sessions\20260918_155836")

def rd(name):
    with (S/name).open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))

raw=rd("raw_100hz.csv")
hits=rd("hits.csv")
stages=rd("stages.csv")

for r in raw:
    r["dt"]=datetime.fromisoformat(r["wall_time"])
    r["pe"]=float(r["protocol_elapsed_s"])
    r["valid"]=int(r["frame_valid"])
    for k in ("ax","ay","az","yaw","pitch","roll"):
        r[k]=float(r[k])
    for k in ("rel_yaw","rel_pitch","rel_roll"):
        try:r[k]=float(r[k])
        except:r[k]=np.nan
    r["mag"]=math.sqrt(r["ax"]**2+r["ay"]**2+r["az"]**2)
    r["stage_index"]=int(r["stage_index"])

for h in hits:
    h["dt"]=datetime.fromisoformat(h["wall_time"])
    h["pe"]=float(h["protocol_elapsed_s"])
    h["prob"]=float(h["prob"])
    h["peak_g"]=float(h["peak_accel_g"])
    h["stage_index"]=int(h["stage_index"])

for s in stages:
    s["stage_index"]=int(s["stage_index"])
    s["start_pe"]=float(s["start_protocol_elapsed_s"])
    s["end_pe"]=float(s["end_protocol_elapsed_s"])
    s["dur"]=s["end_pe"]-s["start_pe"]

print("raw rows visible",len(raw),"hits",len(hits),"stages",len(stages))
print("raw pe range", min(r["pe"] for r in raw), max(r["pe"] for r in raw))

# global packet integrity/hz and invalids
print("\n=== GLOBAL RAW QUALITY ===")
for hand in ("L","R"):
    rr=sorted([r for r in raw if r["hand"]==hand],key=lambda x:x["dt"])
    dts=np.array([(rr[i]["dt"]-rr[i-1]["dt"]).total_seconds() for i in range(1,len(rr))])
    pids=np.array([int(r["packet_id"]) for r in rr])
    dif=np.diff(pids)
    invalid=[r for r in rr if not r["valid"]]
    print(hand,"rows",len(rr),
          "wall_dur",round((rr[-1]["dt"]-rr[0]["dt"]).total_seconds(),3),
          "effective_hz",round((len(rr)-1)/(rr[-1]["dt"]-rr[0]["dt"]).total_seconds(),3),
          "interarrival_ms p50/p95/p99/max",*[round(float(x),3) for x in np.percentile(dts*1000,[50,95,99,100])],
          "packet_gap>1",int(np.sum(dif>1)),"back/dup",int(np.sum(dif<=0)),
          "invalid",len(invalid),Counter(r["invalid_reason"] for r in invalid))

# stage raw/hit summary
print("\n=== STAGE QUALITY + HITS ===")
for s in stages:
    idx=s["stage_index"]; kind=s["kind"]; name=s["name"]
    print(f"\n[{idx:02d}] {kind} {name} dur={s['dur']:.3f}s")
    for hand in ("L","R"):
        rr=[r for r in raw if r["hand"]==hand and r["stage_index"]==idx]
        hh=sorted([h for h in hits if h["hand"]==hand and h["stage_index"]==idx],key=lambda x:x["dt"])
        invalid=[r for r in rr if not r["valid"]]
        if rr:
            hz=len(rr)/s["dur"]
            mags=np.array([r["mag"] for r in rr if r["valid"]])
            motion_pct=float(np.mean(np.abs(mags-1)>0.35)*100) if len(mags) else 0
        else: hz=0; motion_pct=0
        c=Counter(h["pred"] for h in hh)
        top=c.most_common(1)[0] if c else ("—",0)
        stab=top[1]/len(hh) if hh else 0
        intervals=np.array([(hh[i]["dt"]-hh[i-1]["dt"]).total_seconds() for i in range(1,len(hh))])
        short100=int(np.sum(intervals<0.10)) if len(intervals) else 0
        short150=int(np.sum(intervals<0.15)) if len(intervals) else 0
        medint=float(np.median(intervals)) if len(intervals) else None
        print(hand,"raw",len(rr),f"{hz:.1f}Hz","invalid",len(invalid),
              "motion%",round(motion_pct,1),
              "hits",len(hh),"modal",top[0],f"{stab*100:.1f}%",
              "medianProb",round(float(np.median([x["prob"] for x in hh])),3) if hh else None,
              "hitIntMed",round(medint,3) if medint else None,
              "<100ms",short100,"<150ms",short150,
              "labels",dict(c))

# simultaneous stress matching
def pair_hits(L,R,tol):
    L=sorted(L,key=lambda x:x["dt"]);R=sorted(R,key=lambda x:x["dt"])
    used=set();pairs=[]
    for l in L:
        bi=None;bd=None
        for j,r in enumerate(R):
            if j in used:continue
            d=abs((r["dt"]-l["dt"]).total_seconds())
            if d<=tol and (bd is None or d<bd):
                bi=j;bd=d
        if bi is not None:
            used.add(bi);pairs.append((l,R[bi],bd))
    return pairs

print("\n=== STRESS A SIMULTANEOUS PAIRING ===")
L=[h for h in hits if h["stage_index"]==14 and h["hand"]=="L"]
R=[h for h in hits if h["stage_index"]==14 and h["hand"]=="R"]
for tol in (0.03,0.05,0.08,0.12):
    pairs=pair_hits(L,R,tol)
    print("tol",tol,"pairs",len(pairs),"Lpair%",round(len(pairs)/len(L)*100,1),"Rpair%",round(len(pairs)/len(R)*100,1),
          "dt median ms",round(np.median([p[2] for p in pairs])*1000,1) if pairs else None)

# alternating stress hand sequence
print("\n=== STRESS B ALTERNATION ===")
hb=sorted([h for h in hits if h["stage_index"]==16],key=lambda x:x["dt"])
seq="".join(h["hand"] for h in hb)
same=sum(seq[i]==seq[i-1] for i in range(1,len(seq)))
alt=(len(seq)-1-same)/(len(seq)-1) if len(seq)>1 else 0
runs=[]
if seq:
    cur=seq[0];n=1
    for ch in seq[1:]:
        if ch==cur:n+=1
        else:runs.append((cur,n));cur=ch;n=1
    runs.append((cur,n))
print("n",len(hb),"L",sum(h["hand"]=="L" for h in hb),"R",sum(h["hand"]=="R" for h in hb),
      "adjacent alternation rate",round(alt*100,1),"%","max same-hand run",max(n for _,n in runs) if runs else 0)
print("first 120 hand sequence",seq[:120])

# corrupted timing episodes
print("\n=== R INVALID EPISODES ===")
rr=sorted([r for r in raw if r["hand"]=="R" and not r["valid"]],key=lambda x:x["dt"])
eps=[];cur=[]
for r in rr:
    if cur and (r["dt"]-cur[-1]["dt"]).total_seconds()>0.20:
        eps.append(cur);cur=[]
    cur.append(r)
if cur:eps.append(cur)
for e in eps:
    if len(e)>=2:
        print(e[0]["dt"].time(),"->",e[-1]["dt"].time(),"dur",round((e[-1]["dt"]-e[0]["dt"]).total_seconds(),3),
              "n",len(e),"stage",e[0]["stage_index"],e[0]["stage_name"],
              Counter(x["invalid_reason"] for x in e))

# final static stage 17: movement and offline "would detector see hits" later separately
print("\n=== FINAL STATIC RAW ===")
for hand in ("L","R"):
    rr=[r for r in raw if r["hand"]==hand and r["stage_index"]==17]
    valid=[r for r in rr if r["valid"]]
    mags=np.array([r["mag"] for r in valid])
    print(hand,"rows",len(rr),"valid",len(valid),"invalid",len(rr)-len(valid),
          "mag median/p95/max",*[round(float(x),4) for x in np.percentile(mags,[50,95,100])] if len(mags) else [],
          "|mag-1|>0.35%",round(float(np.mean(np.abs(mags-1)>0.35)*100),2) if len(mags) else None,
          "hits_logged",len([h for h in hits if h["hand"]==hand and h["stage_index"]==17]))

# pose at hits, stage cluster summaries
print("\n=== HIT-CENTER RELATIVE POSE CLUSTERS ===")
# nearest raw around hit processing time - 80ms approximate window-center
for idx in [2,4,6,8,10,12]:
    s=[x for x in stages if x["stage_index"]==idx][0]
    print("\n",idx,s["name"])
    for hand in ("L","R"):
        hh=[h for h in hits if h["hand"]==hand and h["stage_index"]==idx]
        rr=[r for r in raw if r["hand"]==hand and r["stage_index"]==idx and r["valid"] and not np.isnan(r["rel_yaw"])]
        vals=[]
        for h in hh:
            target=h["dt"]-timedelta(milliseconds=80)
            if not rr:continue
            n=min(rr,key=lambda r:abs((r["dt"]-target).total_seconds()))
            if abs((n["dt"]-target).total_seconds())<0.05:
                vals.append([n["rel_yaw"],n["rel_pitch"],n["rel_roll"]])
        if vals:
            a=np.array(vals)
            print(hand,"n",len(a),
                  "yaw med/IQR",round(float(np.median(a[:,0])),1),round(float(np.percentile(a[:,0],75)-np.percentile(a[:,0],25)),1),
                  "pitch",round(float(np.median(a[:,1])),1),round(float(np.percentile(a[:,1],75)-np.percentile(a[:,1],25)),1),
                  "roll",round(float(np.median(a[:,2])),1),round(float(np.percentile(a[:,2],75)-np.percentile(a[:,2],25)),1))
