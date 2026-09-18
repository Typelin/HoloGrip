from pathlib import Path
import csv,json
import numpy as np
LAB=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918")
S=LAB/"live_sessions"/"20260918_141423"
REF=json.loads((LAB/"spatial_reference_0826.json").read_text(encoding="utf-8"))
names=REF["feature_names"]
train=np.asarray([r["feature"] for r in REF["events"]],float)
train_h=np.asarray([r["hand"] for r in REF["events"]])
with (S/"hits.csv").open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
# final calibration elapsed
ev=[json.loads(x) for x in (S/"events.jsonl").read_text(encoding="utf-8").splitlines()]
last_cal=max(float(x["elapsed_s"]) for x in ev if x.get("kind")=="CALIBRATE_OK")
rows=[r for r in rows if float(r["elapsed_s"])>=last_cal]
cur=np.asarray([[float(r[n]) for n in names] for r in rows],float)
cur_h=np.asarray([r["hand"] for r in rows])
print("features",names)
for h in ["L","R"]:
    tr=train[train_h==h]
    cu=cur[cur_h==h]
    print("\n===",h,"train",len(tr),"current",len(cu),"===")
    out=[]
    for j,n in enumerate(names):
        tm=np.median(tr[:,j]); cm=np.median(cu[:,j])
        q10,q90=np.percentile(tr[:,j],[10,90])
        robust=max((q90-q10)/2,1e-6)
        shift=(cm-tm)/robust
        outside=np.mean((cu[:,j]<q10)|(cu[:,j]>q90))*100
        out.append((abs(shift),n,tm,q10,q90,cm,shift,outside))
    for a,n,tm,q10,q90,cm,sh,outside in sorted(out,reverse=True):
        print(f"{n:14s} train_med={tm:+8.2f} p10-90=({q10:+8.2f},{q90:+8.2f}) current_med={cm:+8.2f} shift={sh:+6.2f}x outside={outside:5.1f}%")
