from pathlib import Path
import csv,sys,json,random
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
from product_hit_and_zone import ZONE_MAP

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
X5=[];y5=[];h5=[]
for r in readcsv(GT5):
    hh=r["assigned_hand"];d=r["zone_name"]
    if hh not in ("L","R") or d not in ZONE_MAP:continue
    ft=feature_from_samples(D5[hh],float(r["aligned_csv_time_ms"]),80)
    if ft is not None:X5.append(ft);y5.append(ZONE_MAP[d]);h5.append(hh)
X5=np.asarray(X5);y5=np.asarray(y5);h5=np.asarray(h5)

def make(seed=42):
    return Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=1800,random_state=seed))])

# base predictions once
base=np.full(len(y5),-1,int)
for hh in ("L","R"):
    tr=np.where(h==hh)[0];te=np.where(h5==hh)[0]
    m=make(42);m.fit(X[tr],y[tr]);base[te]=m.predict(X5[te])

eligible=[]
for hh in ("L","R"):
    for c in sorted(set(y5[h5==hh])):
        ix=np.where((h5==hh)&(y5==c))[0]
        if len(ix)>=3:
            eligible.append((hh,c,ix))
print("eligible_pairs",[(hh,int(c),len(ix)) for hh,c,ix in eligible])

rows=[]
for seed in range(30):
    rng=np.random.default_rng(seed)
    calib=[]
    for hh,c,ix in eligible:
        calib.append(int(rng.choice(ix)))
    calib=np.array(sorted(set(calib)),int)
    test=np.array([i for i in range(len(y5)) if i not in set(calib)],int)
    pred=np.full(len(test),-1,int)
    for hh in ("L","R"):
        tr=np.where(h==hh)[0]
        calh=calib[h5[calib]==hh]
        pos=np.where(h5[test]==hh)[0]
        teidx=test[pos]
        Xtr=np.vstack([X[tr],X5[calh]]) if len(calh) else X[tr]
        ytr=np.concatenate([y[tr],y5[calh]]) if len(calh) else y[tr]
        m=make(42);m.fit(Xtr,ytr);pred[pos]=m.predict(X5[teidx])
    yy=y5[test];bb=base[test]
    row={
        "seed":seed,
        "adapt_acc":float(accuracy_score(yy,pred)),
        "adapt_bal":float(balanced_accuracy_score(yy,pred)),
        "adapt_f1":float(f1_score(yy,pred,average="macro")),
        "base_acc":float(accuracy_score(yy,bb)),
        "base_bal":float(balanced_accuracy_score(yy,bb)),
        "base_f1":float(f1_score(yy,bb,average="macro")),
    }
    row["delta_acc"]=row["adapt_acc"]-row["base_acc"]
    row["delta_bal"]=row["adapt_bal"]-row["base_bal"]
    row["delta_f1"]=row["adapt_f1"]-row["base_f1"]
    rows.append(row)
    print(seed,row)

print("\nSUMMARY")
for key in ("adapt_acc","adapt_bal","adapt_f1","delta_acc","delta_bal","delta_f1"):
    a=np.array([r[key] for r in rows])
    print(key,"mean",a.mean(),"median",np.median(a),"min",a.min(),"p10",np.percentile(a,10),"p90",np.percentile(a,90),"max",a.max())
print("improve acc",sum(r["delta_acc"]>0 for r in rows),"/",len(rows))
print("improve bal",sum(r["delta_bal"]>0 for r in rows),"/",len(rows))
print("improve f1",sum(r["delta_f1"]>0 for r in rows),"/",len(rows))
(LAB/"fewshot_random_robustness.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
