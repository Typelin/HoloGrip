from pathlib import Path
import csv,sys,json
from collections import defaultdict
import numpy as np
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM"; sys.path.insert(0,str(APP))
from product_hit_and_zone import load_raw_rows, rows_to_samples, slice_window, window_features, FEATURE_NAMES

# 0805 high-confidence events
raw5=rows_to_samples(load_raw_rows(str(ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv")))
with (ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv").open(encoding="utf-8-sig",newline="") as f:
    e5=list(csv.DictReader(f))
D5=defaultdict(list)
for e in e5:
    h=e["assigned_hand"]; d=e["zone_name"]; c=float(e["aligned_csv_time_ms"])
    ft=window_features(slice_window(raw5[h],c,80.0),c)
    if ft is not None:D5[d].append(ft)

# 0812 final GT
raw12=rows_to_samples(load_raw_rows(str(ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv")))
with (ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv").open(encoding="utf-8",newline="") as f:
    e12=list(csv.DictReader(f))
D12=defaultdict(list)
for e in e12:
    h=e["final_hand"]; d=e["drum"]
    if h not in ("L","R"):continue
    c=float(e["csv_center_ms"])
    ft=window_features(slice_window(raw12[h],c,80.0),c)
    if ft is not None:D12[d].append(ft)

aliases={"小鼓":"Snare","高音 Tom":"HighTom","中音 Tom":"MidTom","落地 Tom":"FloorTom","Hi-Hat":"HiHat","Crash":"Crash","Ride":"Ride"}
sel=["center_yaw","center_pitch","mean_roll","std_yaw","yaw_range","max_ax","max_ay","max_az"]
idx={n:FEATURE_NAMES.index(n) for n in sel}
for d in ["Hi-Hat","小鼓","落地 Tom","Ride","Crash","高音 Tom","中音 Tom"]:
    a=np.asarray(D5.get(d,[]),float); b=np.asarray(D12.get(d,[]),float)
    print("\n==",aliases[d],"0805",len(a),"0812",len(b),"==")
    if len(a)==0 or len(b)==0: continue
    for n in sel:
        j=idx[n]
        m5=float(np.median(a[:,j])); m12=float(np.median(b[:,j]))
        q10,q90=np.percentile(b[:,j],[10,90]); scale=max((q90-q10)/2,1e-6)
        print(f"{n:12s} 0805={m5:+8.2f} 0812={m12:+8.2f} delta={m5-m12:+8.2f} shift_vs_0812={((m5-m12)/scale):+6.2f}x")
