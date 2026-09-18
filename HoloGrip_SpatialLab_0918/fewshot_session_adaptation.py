from pathlib import Path
import csv,sys,json
from collections import Counter,defaultdict
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(LAB));sys.path.insert(0,str(APP))
from vnext_core import compute_r0,make_relative_sample,feature_from_samples
from product_hit_and_zone import ZONE_MAP,ZONE_NAMES

RAW12=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT12=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
RAW5=ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"
GT5=ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv"

def readcsv(p,enc="utf-8-sig"):
    with p.open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
def prep(rawp):
    rows=readcsv(rawp);out={}
    for h in ("L","R"):
        rr=[r for r in rows if r["hand"]==h]
        t0=min(float(r["song_time_ms"]) for r in rr)
        cal=[r for r in rr if float(r["song_time_ms"])<t0+1000]
        R0,_=compute_r0([(float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in cal])
        out[h]=[make_relative_sample(R0,float(r["song_time_ms"]),float(r["ax_g"]),float(r["ay_g"]),float(r["az_g"]),
                    float(r["yaw_deg"]),float(r["pitch_deg"]),float(r["roll_deg"])) for r in rr]
    return out
D12=prep(RAW12);D5=prep(RAW5)

X=[];y=[];h=[]
for r in readcsv(GT12,"utf-8"):
    hh=r["final_hand"];d=r["drum"]
    if hh not in ("L","R") or d not in ZONE_MAP:continue
    ft=feature_from_samples(D12[hh],float(r["csv_center_ms"]),80)
    if ft is not None:X.append(ft);y.append(ZONE_MAP[d]);h.append(hh)
X=np.asarray(X);y=np.asarray(y);h=np.asarray(h)

X5=[];y5=[];h5=[];t5=[];meta=[]
for r in readcsv(GT5):
    hh=r["assigned_hand"];d=r["zone_name"]
    if hh not in ("L","R") or d not in ZONE_MAP:continue
    ft=feature_from_samples(D5[hh],float(r["aligned_csv_time_ms"]),80)
    if ft is not None:
        X5.append(ft);y5.append(ZONE_MAP[d]);h5.append(hh);t5.append(float(r["aligned_csv_time_ms"]));meta.append(r)
X5=np.asarray(X5);y5=np.asarray(y5);h5=np.asarray(h5);t5=np.asarray(t5)

def make():
    return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1800,random_state=42))])

def metrics(yy,pp):
    return {
        "n":int(len(yy)),
        "acc":float(accuracy_score(yy,pp)) if len(yy) else 0,
        "bal":float(balanced_accuracy_score(yy,pp)) if len(np.unique(yy))>1 else 0,
        "f1":float(f1_score(yy,pp,average="macro")) if len(yy) else 0,
    }

# base
basepred=np.full(len(y5),-1,int)
for hh in ("L","R"):
    tr=np.where(h==hh)[0];te=np.where(h5==hh)[0]
    m=make();m.fit(X[tr],y[tr]);basepred[te]=m.predict(X5[te])
print("BASE ALL",metrics(y5,basepred))

results={}
for N in (1,2,3,5):
  for repeat in (1,3,5,10):
    calib=[];test=[]
    # first N chronological examples for each hand+class, only if class has at least N+2
    for hh in ("L","R"):
      for c in sorted(set(y5[h5==hh])):
        ix=np.where((h5==hh)&(y5==c))[0]
        ix=ix[np.argsort(t5[ix])]
        if len(ix)>=N+2:
            calib.extend(ix[:N].tolist());test.extend(ix[N:].tolist())
        else:
            # no calibration possible; retain all in test using base model knowledge
            test.extend(ix.tolist())
    calib=np.array(sorted(set(calib)),int); test=np.array(sorted(set(test)),int)
    pred=np.full(len(test),-1,int)
    for hh in ("L","R"):
        trbase=np.where(h==hh)[0]
        calh=calib[h5[calib]==hh] if len(calib) else np.array([],int)
        teh=np.where(h5[test]==hh)[0]
        teidx=test[teh]
        Xtr=[X[trbase]];ytr=[y[trbase]]
        if len(calh):
            Xtr += [np.repeat(X5[calh],repeat,axis=0)]
            ytr += [np.repeat(y5[calh],repeat,axis=0)]
        m=make();m.fit(np.vstack(Xtr),np.concatenate(ytr))
        pred[teh]=m.predict(X5[teidx])
    mm=metrics(y5[test],pred)
    # baseline on same remaining test set
    bm=metrics(y5[test],basepred[test])
    key=f"N{N}_x{repeat}"
    results[key]={"adapt":mm,"base_same_test":bm,"calib_n":int(len(calib)),"test_n":int(len(test))}
    print(key,"calib",len(calib),"test",len(test),"BASE",bm,"ADAPT",mm)

print("\n=== RANK ===")
for k,v in sorted(results.items(),key=lambda kv:kv[1]["adapt"]["bal"]+0.2*kv[1]["adapt"]["acc"],reverse=True):
    print(k,v)

(LAB/"fewshot_adaptation_results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
