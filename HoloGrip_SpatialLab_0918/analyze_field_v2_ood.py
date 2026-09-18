from pathlib import Path
import csv,sys
from collections import Counter,defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples,FEATURE_NAMES,ZONE_NAMES

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
LIVE=LAB/"field_v2_sessions/20260918_210839/resolved_hits.csv"
Z={n:i for i,n in enumerate(ZONE_NAMES)}

def rcsv(p,enc="utf-8-sig"):
    with Path(p).open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
rows=rcsv(RAW);D={}
for h in ("L","R"):
    rr=[r for r in rows if r["hand"]==h]
    t0=min(float(r["song_time_ms"]) for r in rr)
    cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
    R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
    D[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
        float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
train={"L":[],"R":[]};tlab={"L":[],"R":[]}
for g in rcsv(GT,"utf-8"):
    h=g["final_hand"]
    if h not in train or g["drum"] not in Z:continue
    t=float(g["csv_center_ms"])
    for o in (-20,0,20):
        ft=feature_from_samples(D[h],t+o,80)
        if ft is not None:
            train[h].append(ft);tlab[h].append(g["drum"])
live=rcsv(LIVE,"utf-8")
print("LIVE N",len(live))
for h in ("L","R"):
    T=np.asarray(train[h],float)
    sc=StandardScaler().fit(T)
    ZT=sc.transform(T)
    nn=NearestNeighbors(n_neighbors=2).fit(ZT)
    # training leave-self NN thresholds
    dtrain=nn.kneighbors(ZT,return_distance=True)[0][:,1]
    p95,p99=np.percentile(dtrain,[95,99])
    print("\nHAND",h,"train",len(T),"NN p95/p99",round(p95,2),round(p99,2))
    q=[r for r in live if r["hand"]==h]
    if not q:continue
    X=np.asarray([[float(r[n]) for n in FEATURE_NAMES] for r in q])
    ZX=sc.transform(X)
    dl,ii=nn.kneighbors(ZX,n_neighbors=1,return_distance=True)
    dl=dl[:,0];ii=ii[:,0]
    print("live NN median/p90/max",np.round(np.percentile(dl,[50,90,100]),2),
          "IN",int(np.sum(dl<=p95)),"border",int(np.sum((dl>p95)&(dl<=p99))),"OOD",int(np.sum(dl>p99)))
    by=defaultdict(list)
    for r,d,i in zip(q,dl,ii):
        by[r["marker"]].append((r,d,tlab[h][i]))
    for marker,items in by.items():
        ds=np.array([x[1] for x in items])
        print(" marker",marker,"n",len(items),"med/p90",np.round(np.percentile(ds,[50,90]),2),
              "IN/OOD",int(np.sum(ds<=p95)),int(np.sum(ds>p99)),
              "pred",dict(Counter(x[0]["raw_drum"] for x in items)),
              "nearest_train",dict(Counter(x[2] for x in items)))
