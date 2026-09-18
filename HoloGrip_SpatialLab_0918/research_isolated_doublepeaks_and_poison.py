from pathlib import Path
import csv,sys,math
from collections import Counter,defaultdict
import numpy as np

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
LAB=ROOT/"HoloGrip_SpatialLab_0918"
APP=ROOT/"Apps"/"Song_Collection_COM"
sys.path.insert(0,str(APP))
from song_collection_server import HitDetector,SensorPacket
from product_hit_and_zone import PRODUCT_DETECTOR_KWARGS,PEAK_LAG_MS

RAW=ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"
GT=ROOT/"Data/Derived/Song_Collection_COM/S20260812_P01_song02_T1127_cleaned/Song2_ground_truth_labels_final_0821.csv"

def rcsv(p,enc="utf-8-sig"):
    with Path(p).open(encoding=enc,newline="") as f:return list(csv.DictReader(f))
rows=rcsv(RAW)
gt=[r for r in rcsv(GT,"utf-8") if r["final_hand"] in ("L","R")]

# detector replay
det={h:HitDetector(**PRODUCT_DETECTOR_KWARGS) for h in ("L","R")}
ds=[]
for r in rows:
    h=r["hand"];t=float(r["song_time_ms"])
    pkt=SensorPacket(hand=h,ax=float(r["ax_g"]),ay=float(r["ay_g"]),az=float(r["az_g"]),
        yaw=float(r["yaw_deg"]),pitch=float(r["pitch_deg"]),roll=float(r["roll_deg"]),
        packet_id=int(r["packet_id"]),sensor_time_ms=int(float(r["sensor_time_ms"])),
        received_time_ms=int(float(r["received_time_ms"])),received_monotonic=t/1000)
    ev=det[h].add_packet(pkt,float(r["cal_yaw_deg"]),float(r["cal_pitch_deg"]))
    if ev:ds.append({"h":h,"t":t-PEAK_LAG_MS,"g":float(ev["peak_accel_g"])})

# same-hand GT isolation distance
byh={h:sorted([float(g["csv_center_ms"]) for g in gt if g["final_hand"]==h]) for h in ("L","R")}
iso=[]
for g in gt:
    h=g["final_hand"];t=float(g["csv_center_ms"])
    near=[abs(x-t) for x in byh[h] if abs(x-t)>1e-6]
    mind=min(near) if near else 9999
    if mind>=300: iso.append((g,mind))
print("isolated GT >=300ms same-hand nearest:",len(iso),Counter(g["final_hand"] for g,_ in iso),Counter(g["drum"] for g,_ in iso))

multi=[]
for g,mind in iso:
    h=g["final_hand"];t=float(g["csv_center_ms"])
    q=[d for d in ds if d["h"]==h and abs(d["t"]-t)<=200]
    q=sorted(q,key=lambda x:x["t"])
    if len(q)>=2:
        multi.append((g,q))
print("isolated GT with >=2 detections within +/-200ms:",len(multi))
gapc=Counter()
nearer_first=nearer_last=0
g_relation=[]
for g,q in multi:
    t=float(g["csv_center_ms"])
    # consider adjacent pair closest to GT if >2
    pair=min([(q[i],q[i+1]) for i in range(len(q)-1)],key=lambda z:min(abs(z[0]["t"]-t),abs(z[1]["t"]-t)))
    a,b=pair;gap=b["t"]-a["t"];gapc[round(gap/10)*10]+=1
    if abs(a["t"]-t)<=abs(b["t"]-t):nearer_first+=1
    else:nearer_last+=1
    g_relation.append((b["g"]/max(a["g"],1e-9),a["t"]-t,b["t"]-t,gap,g["drum"],g["final_hand"]))
print("pair gap bins",gapc)
print("closer to GT first/second",nearer_first,nearer_last)
if g_relation:
    arr=np.array([[x[0],x[1],x[2],x[3]] for x in g_relation],float)
    print("second/first g ratio median p10/90",np.round(np.percentile(arr[:,0],[10,50,90]),2))
    print("first offset p10/50/90",np.round(np.percentile(arr[:,1],[10,50,90]),1))
    print("second offset p10/50/90",np.round(np.percentile(arr[:,2],[10,50,90]),1))
    print("gap p10/50/90",np.round(np.percentile(arr[:,3],[10,50,90]),1))

print("\nexamples:")
for g,q in multi[:30]:
    t=float(g["csv_center_ms"])
    print(g["event_id"],g["final_hand"],g["drum"],"GT",round(t,1),
          [(round(x["t"]-t,1),round(x["g"],2)) for x in q])

# Historical right data poison scan for 8/12 and 8/5 via raw pattern rules, no FrameValidator import.
def reason(p,last_sig,sig_repeat):
    vals=[p[k] for k in ("ax","ay","az","yaw","pitch","roll")]
    if not all(math.isfinite(v) for v in vals):return False,"non_finite",last_sig,sig_repeat
    if all(abs(v)<1e-12 for v in vals):return False,"all_zero",last_sig,sig_repeat
    acc_same=max(abs(p["ax"]-p["ay"]),abs(p["ay"]-p["az"]),abs(p["ax"]-p["az"]))<1e-6
    ang_same=max(abs(p["yaw"]-p["pitch"]),abs(p["pitch"]-p["roll"]),abs(p["yaw"]-p["roll"]))<1e-6
    scale_same=abs(p["yaw"]-p["ax"]*11.25)<0.02
    if acc_same and ang_same and scale_same and abs(p["ax"])>0.01:return False,"i2c_same_word",last_sig,sig_repeat
    mag=math.sqrt(p["ax"]**2+p["ay"]**2+p["az"]**2)
    sig=tuple(round(p[k],4) for k in ("ax","ay","az","yaw","pitch","roll"))
    sr=sig_repeat+1 if sig==last_sig else 1
    if sr>=5 and mag>2.3:return False,"stale_high_plateau",sig,sr
    return True,"",sig,sr

for rawp in [
    RAW,
    ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"
]:
    rr=rcsv(rawp)
    print("\nPOISON",rawp.name)
    for h in ("L","R"):
        last=None;sr=0;c=Counter();n=0
        for r in rr:
            if r["hand"]!=h:continue
            p={"ax":float(r["ax_g"]),"ay":float(r["ay_g"]),"az":float(r["az_g"]),
               "yaw":float(r["yaw_deg"]),"pitch":float(r["pitch_deg"]),"roll":float(r["roll_deg"])}
            ok,rs,last,sr=reason(p,last,sr);n+=1
            if not ok:c[rs]+=1
        print(h,"n",n,"bad",sum(c.values()),"pct",round(100*sum(c.values())/max(1,n),4),dict(c))
