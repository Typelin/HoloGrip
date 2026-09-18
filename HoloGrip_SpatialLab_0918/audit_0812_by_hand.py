from pathlib import Path
import csv,sys
from collections import Counter,defaultdict
import numpy as np

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM"; sys.path.insert(0,str(APP))
from product_hit_and_zone import load_raw_rows,rows_to_samples,slice_window,window_features,FEATURE_NAMES

raw=rows_to_samples(load_raw_rows(str(ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv")))
gtp=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"
with gtp.open(encoding="utf-8",newline="") as f:gt=list(csv.DictReader(f))

by=defaultdict(list)
for r in gt:
    h=r["final_hand"]; d=r["drum"]
    if h not in ("L","R"):continue
    c=float(r["csv_center_ms"])
    ft=window_features(slice_window(raw[h],c,80.0),c)
    if ft is not None: by[(d,h)].append(ft)

print("HAND COUNTS")
for d in ["小鼓","高音 Tom","中音 Tom","落地 Tom","Hi-Hat","Crash","Ride"]:
    print(d,"L",len(by[(d,"L")]),"R",len(by[(d,"R")]))

for d in ["小鼓","高音 Tom","中音 Tom","落地 Tom","Hi-Hat","Crash","Ride"]:
    a=np.asarray(by[(d,"L")],float); b=np.asarray(by[(d,"R")],float)
    print("\n==",d,"==")
    if len(a)==0 or len(b)==0:
        print("only one hand")
        continue
    for n in ["center_yaw","center_pitch","mean_yaw","mean_pitch","mean_roll","std_yaw","yaw_range","max_ax","max_ay","max_az"]:
        j=FEATURE_NAMES.index(n)
        ml=float(np.median(a[:,j])); mr=float(np.median(b[:,j]))
        print(f"{n:12s} L={ml:+8.2f} R={mr:+8.2f} deltaR-L={mr-ml:+8.2f}")
