from pathlib import Path
import csv,sys
from collections import defaultdict
import numpy as np
ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
APP=ROOT/"Apps"/"Song_Collection_COM";sys.path.insert(0,str(APP))
from product_hit_and_zone import load_raw_rows,rows_to_samples,slice_window,window_features,FEATURE_NAMES
def collect(rawp,eventp,session):
    raw=rows_to_samples(load_raw_rows(str(rawp)))
    with eventp.open(encoding="utf-8-sig" if session=="5" else "utf-8",newline="") as f:ev=list(csv.DictReader(f))
    by=defaultdict(list)
    for r in ev:
        if session=="5":
            h=r["assigned_hand"];d=r["zone_name"];c=float(r["aligned_csv_time_ms"])
        else:
            h=r["final_hand"];d=r["drum"];c=float(r["csv_center_ms"])
        if h not in ("L","R"):continue
        ft=window_features(slice_window(raw[h],c,80),c)
        if ft is not None:by[(d,h)].append(np.asarray(ft))
    return by
b5=collect(ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv",ROOT/"Data/Derived/Song_Collection_COM/S20260805_P01_song01/hand_label_triage_0807/auto_accepted_single_events.csv","5")
b12=collect(ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv",ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv","12")
for h in ("L","R"):
 print("\nHAND",h)
 for d in ["Hi-Hat","Crash","落地 Tom","Ride","小鼓","高音 Tom"]:
  a=np.asarray(b5[(d,h)],float);b=np.asarray(b12[(d,h)],float)
  if len(a)==0 and len(b)==0:continue
  print(d,"0805",len(a),"0812",len(b))
  for nm in ["center_yaw","center_pitch","mean_roll","max_mag","std_yaw","yaw_range"]:
   j=FEATURE_NAMES.index(nm)
   s5="NA" if len(a)==0 else f"{np.median(a[:,j]):+.1f}"
   s12="NA" if len(b)==0 else f"{np.median(b[:,j]):+.1f}"
   print(" ",nm,s5,s12)
