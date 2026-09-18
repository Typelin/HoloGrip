from pathlib import Path
import csv, sys, json
from collections import Counter, defaultdict
import numpy as np, joblib

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(APP))
from product_hit_and_zone import load_raw_rows, rows_to_samples, slice_window, window_features, predict_zone

raw_path=ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"
events_path=ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv"
model_path=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/hologrip_song2_七鼓點模型_真正驗證版_0826.joblib"

raw=load_raw_rows(str(raw_path))
samples=rows_to_samples(raw)
clf=joblib.load(model_path)

with events_path.open(encoding="utf-8-sig",newline="") as f:
    events=list(csv.DictReader(f))

rows=[]
for e in events:
    h=e["assigned_hand"]
    true=e["zone_name"]
    c=float(e["aligned_csv_time_ms"])
    if h not in ("L","R"): continue
    feat=window_features(slice_window(samples[h],c,80.0),c)
    if feat is None: continue
    pred,prob=predict_zone(clf,feat)
    rows.append({
        "event_id":int(e["event_id"]),"hand":h,"true":true,"pred":pred,"prob":prob,
        "dominant_share":float(e["dominant_hand_share"]),"peak":float(e["dominant_peak_g"]),
        "feature":feat
    })

print("events",len(rows))
ok=sum(r["true"]==r["pred"] for r in rows)
print("accuracy",ok/len(rows) if rows else 0,ok,"/",len(rows))
print("true",Counter(r["true"] for r in rows))
print("pred",Counter(r["pred"] for r in rows))
for d in sorted(set(r["true"] for r in rows)):
    rr=[r for r in rows if r["true"]==d]
    c=sum(r["true"]==r["pred"] for r in rr)
    print(d,c,"/",len(rr),"acc",round(c/len(rr),3),"preds",Counter(r["pred"] for r in rr))

# confidence-conditioned
for th in [0.90,0.95,0.97,0.98]:
    rr=[r for r in rows if r["dominant_share"]>=th]
    if rr:
        c=sum(r["true"]==r["pred"] for r in rr)
        print("share>=",th,"n",len(rr),"acc",round(c/len(rr),3))

# OOD using 8/12 reference scaler nearest-neighbor thresholds from spatial lab
ref=json.loads((ROOT/"HoloGrip_SpatialLab_0918/spatial_reference_0826.json").read_text(encoding="utf-8"))
Xref=np.asarray([x["feature"] for x in ref["events"]],float)
scaler=clf.named_steps.get("scaler")
Zref=scaler.transform(Xref)
# thresholds leave-one-out NN among training events
D=np.linalg.norm(Zref[:,None,:]-Zref[None,:,:],axis=2)
np.fill_diagonal(D,np.inf)
train_nn=np.min(D,axis=1)
p95,p99=np.percentile(train_nn,[95,99])
print("NN thresholds",p95,p99)
X=np.asarray([r["feature"] for r in rows],float)
Z=scaler.transform(X)
dist=np.min(np.linalg.norm(Z[:,None,:]-Zref[None,:,:],axis=2),axis=1)
bands=np.where(dist<=p95,"IN-LIKE",np.where(dist<=p99,"BORDERLINE","OOD"))
print("bands",Counter(bands.tolist()),"OOD_pct",round(float(np.mean(bands=="OOD")*100),1))
print("NN median/p95",float(np.median(dist)),float(np.percentile(dist,95)))

out=ROOT/"HoloGrip_SpatialLab_0918"/"cross_session_0805_using_0812_model.csv"
with out.open("w",encoding="utf-8",newline="") as f:
    w=csv.writer(f)
    w.writerow(["event_id","hand","true","pred","prob","dominant_share","peak_g","nn_distance","band"])
    for r,d,b in zip(rows,dist,bands):
        w.writerow([r["event_id"],r["hand"],r["true"],r["pred"],r["prob"],r["dominant_share"],r["peak"],d,b])
print("saved",out)
