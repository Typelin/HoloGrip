from pathlib import Path
import csv,sys
from collections import defaultdict
import numpy as np
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM";sys.path.insert(0,str(APP))
from product_hit_and_zone import load_raw_rows,rows_to_samples,slice_window,window_features,FEATURE_NAMES
raw=rows_to_samples(load_raw_rows(str(ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv")))
with (ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv").open(encoding="utf-8",newline="") as f:gt=list(csv.DictReader(f))
by=defaultdict(list)
for r in gt:
    h=r["final_hand"];d=r["drum"]
    if h not in ("L","R"):continue
    c=float(r["csv_center_ms"]);ft=window_features(slice_window(raw[h],c,80),c)
    if ft is not None:by[(d,h)].append((c,np.asarray(ft)))
for d in ["小鼓","高音 Tom","中音 Tom","落地 Tom","Hi-Hat","Crash","Ride"]:
  for h in ("L","R"):
    arr=sorted(by[(d,h)])
    if len(arr)<8:continue
    n=max(3,len(arr)//3)
    early=np.vstack([x[1] for x in arr[:n]]);late=np.vstack([x[1] for x in arr[-n:]])
    print("\n",d,h,"n",len(arr),"early",n,"late",n,"time",round(arr[0][0]/1000,1),"->",round(arr[-1][0]/1000,1))
    for nm in ["center_yaw","center_pitch","mean_roll","std_yaw","yaw_range"]:
        j=FEATURE_NAMES.index(nm);a=float(np.median(early[:,j]));b=float(np.median(late[:,j]))
        dy=((b-a+180)%360)-180 if "yaw" in nm and nm not in ("std_yaw","yaw_range") else b-a
        print(f" {nm:12s} early={a:+7.2f} late={b:+7.2f} delta={dy:+7.2f}")
