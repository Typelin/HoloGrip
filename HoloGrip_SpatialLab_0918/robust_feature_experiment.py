from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(APP))
from product_hit_and_zone import load_gt, load_raw_rows, rows_to_samples, GT_CSV_PATH, RAW_CSV_PATH, ZONE_MAP, ZONE_NAMES

LAB=ROOT/"HoloGrip_SpatialLab_0918"
SESSION=LAB/"live_sessions"/"20260918_141423"
BLOCK_MS=4000.0
HALF_MS=80.0
SEED=42

FEATURES=[
    "max_mag","mean_mag","std_mag","energy","jerk_max",
    "mean_yaw_sin","mean_yaw_cos","mean_pitch",
    "std_yaw_local","std_pitch",
    "center_yaw_sin","center_yaw_cos","center_pitch",
    "swing_depth","yaw_range_local","hand_flag",
]

def circ_mean_deg(a):
    r=np.deg2rad(np.asarray(a,float))
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(r)),np.mean(np.cos(r)))))

def robust_feat(samples, center_t, hand):
    if not samples: return None
    mags=np.asarray([s["mag"] for s in samples],float)
    yaw=np.asarray([s["yaw"] for s in samples],float)
    pitch=np.asarray([s["pitch"] for s in samples],float)
    yu=np.rad2deg(np.unwrap(np.deg2rad(yaw)))
    nearest=min(samples,key=lambda s:abs(s["song_time_ms"]-center_t))
    my=circ_mean_deg(yaw)
    cy=float(nearest["yaw"])
    return np.asarray([
        np.max(mags),np.mean(mags),np.std(mags),np.sum(np.abs(mags-1.0)),
        np.max(np.abs(np.diff(mags))) if len(mags)>1 else 0.0,
        math.sin(math.radians(my)),math.cos(math.radians(my)),np.mean(pitch),
        np.std(yu),np.std(pitch),
        math.sin(math.radians(cy)),math.cos(math.radians(cy)),float(nearest["pitch"]),
        np.max(pitch)-np.min(pitch),np.max(yu)-np.min(yu),
        1.0 if hand=="R" else -1.0,
    ],float)

def slice_old(samples,center):
    return [s for s in samples if center-HALF_MS <= s["song_time_ms"] <= center+HALF_MS]

# Original 8/12 training data
gt=load_gt(GT_CSV_PATH)
raw_rows=load_raw_rows(RAW_CSV_PATH)
samples=rows_to_samples(raw_rows)
X=[]; y=[]; groups=[]
for r in gt:
    hand=r["final_hand"]; drum=r["drum"]
    if hand not in ("L","R") or drum not in ZONE_MAP: continue
    c=float(r["csv_center_ms"])
    f=robust_feat(slice_old(samples[hand],c),c,hand)
    if f is None: continue
    X.append(f); y.append(ZONE_MAP[drum]); groups.append(int(c//BLOCK_MS))
X=np.asarray(X); y=np.asarray(y); groups=np.asarray(groups)
print("TRAIN",X.shape)

# same 4s-group CV
cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
pred=np.full(len(y),-1,int)
for tr,te in cv.split(X,y,groups):
    m=Pipeline([("scaler",StandardScaler()),("mlp",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1600,random_state=SEED))])
    m.fit(X[tr],y[tr]); pred[te]=m.predict(X[te])
print("CV accuracy",accuracy_score(y,pred),"balanced",balanced_accuracy_score(y,pred),"macroF1",f1_score(y,pred,average="macro"))

# fit final
model=Pipeline([("scaler",StandardScaler()),("mlp",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1600,random_state=SEED))])
model.fit(X,y)
Z=model.named_steps["scaler"].transform(X)
D=np.linalg.norm(Z[:,None,:]-Z[None,:,:],axis=2)
np.fill_diagonal(D,np.inf)
nn=np.min(D,axis=1)
nn95=float(np.percentile(nn,95)); nn99=float(np.percentile(nn,99))
print("NN thresholds",nn95,nn99)

# current raw after final calibration
with (SESSION/"raw_packets.csv").open(encoding="utf-8",newline="") as f:
    rr=list(csv.DictReader(f))
with (SESSION/"events.jsonl").open(encoding="utf-8") as f:
    ev=[json.loads(x) for x in f if x.strip()]
last_cal=max(float(o["elapsed_s"]) for o in ev if o.get("kind")=="CALIBRATE_OK")
print("last_cal",last_cal)

# live raw into per-hand samples using elapsed ms as time
live={"L":[],"R":[]}
for r in rr:
    if not r["cal_yaw"] or float(r["elapsed_s"])<last_cal: continue
    h=r["hand"]
    live[h].append({
        "song_time_ms":float(r["elapsed_s"])*1000.0,
        "mag":float(r["mag_g"]),
        "yaw":float(r["cal_yaw"]),
        "pitch":float(r["cal_pitch"]),
    })

with (SESSION/"hits.csv").open(encoding="utf-8",newline="") as f:
    hits=[r for r in csv.DictReader(f) if float(r["elapsed_s"])>=last_cal]

curX=[]; curmeta=[]
for r in hits:
    h=r["hand"]
    # hit emitted ~80 ms after detected peak
    c=(float(r["elapsed_s"])-0.080)*1000.0
    sl=[s for s in live[h] if c-HALF_MS <= s["song_time_ms"] <= c+HALF_MS]
    ft=robust_feat(sl,c,h)
    if ft is None: continue
    curX.append(ft); curmeta.append(r)
curX=np.asarray(curX)
CZ=model.named_steps["scaler"].transform(curX)
dist=np.linalg.norm(CZ[:,None,:]-Z[None,:,:],axis=2)
nnd=np.min(dist,axis=1)
bands=np.where(nnd<=nn95,"IN-LIKE",np.where(nnd<=nn99,"BORDERLINE","OOD"))
pp=model.predict(curX)
print("CURRENT",len(curX),"bands",Counter(bands.tolist()),"OOD%",100*np.mean(bands=="OOD"))
print("CURRENT pred",Counter(ZONE_NAMES[int(i)] for i in pp))

for h in ("L","R"):
    idx=[i for i,r in enumerate(curmeta) if r["hand"]==h]
    if not idx: continue
    bb=bands[idx]; pph=pp[idx]
    print(h,"n",len(idx),"bands",Counter(bb.tolist()),"OOD%",100*np.mean(bb=="OOD"),"pred",Counter(ZONE_NAMES[int(i)] for i in pph))

# compare yaw local feature shift
for j,n in enumerate(FEATURES):
    tm=np.median(X[:,j]); cm=np.median(curX[:,j]); q10,q90=np.percentile(X[:,j],[10,90])
    out=np.mean((curX[:,j]<q10)|(curX[:,j]>q90))*100
    print(f"{n:16s} train_med={tm:+8.3f} cur_med={cm:+8.3f} train10-90=({q10:+8.3f},{q90:+8.3f}) outside={out:5.1f}%")
