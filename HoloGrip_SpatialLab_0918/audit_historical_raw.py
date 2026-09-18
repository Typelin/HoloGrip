from pathlib import Path
import csv, math, json
from collections import defaultdict
import numpy as np

ROOT=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip")
files=[
("0802_song1", ROOT/"Data/Raw/Song_Collections/S20260802_P01_song01_raw_100hz_20260802_131714.csv"),
("0805_song1", ROOT/"Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv"),
("0812_song1_a", ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song01_raw_100hz_20260812_111808.csv"),
("0812_song1_b", ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song01_raw_100hz_20260812_111817.csv"),
("0812_song2", ROOT/"Data/Raw/Song_Collection_COM/S20260812_P01_song02_raw_100hz_20260812_112101.csv"),
]

def col(row,*names):
    for n in names:
        if n in row and row[n] not in ("",None):
            return row[n]
    return None

def circ_step(a):
    if len(a)<2:return np.array([])
    d=np.diff(a)
    return ((d+180)%360)-180

def summarize(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    out={"rows":len(rows),"header":list(rows[0]) if rows else []}
    by=defaultdict(list)
    for r in rows:
        h=col(r,"hand","HAND_ID","Hand")
        if h in ("L","R"): by[h].append(r)
    for h,rr in by.items():
        def arr(*names):
            vals=[]
            for r in rr:
                x=col(r,*names)
                if x is not None:
                    try: vals.append(float(x))
                    except: pass
            return np.asarray(vals,float)
        yaw=arr("cal_yaw_deg","yaw_deg","yaw")
        pitch=arr("cal_pitch_deg","pitch_deg","pitch")
        roll=arr("roll_deg","roll")
        ax=arr("ax_g","ax"); ay=arr("ay_g","ay"); az=arr("az_g","az")
        mag=arr("accel_magnitude_g","mag_g")
        if len(mag)==0 and len(ax):
            mag=np.sqrt(ax*ax+ay*ay+az*az)
        st=arr("song_time_ms","received_time_ms","sensor_time_ms")
        packet=arr("packet_id")
        s={}
        s["n"]=len(rr)
        if len(st)>1:
            dur=(np.max(st)-np.min(st))
            # received_time is epoch ms, song/sensor is ms too
            s["duration_s"]=float(dur/1000)
            s["row_hz"]=float(len(rr)/(dur/1000)) if dur>0 else None
        for name,a in [("yaw",yaw),("pitch",pitch),("roll",roll),("mag",mag)]:
            if len(a):
                s[name+"_p01_p50_p99"]=[float(x) for x in np.percentile(a,[1,50,99])]
        if len(yaw)>1:
            ds=np.abs(circ_step(yaw))
            s["yaw_step_abs_p50_p95_p99"]=[float(x) for x in np.percentile(ds,[50,95,99])]
            s["raw_wrap_crossings"]=int(np.sum(np.abs(np.diff(yaw))>300))
        if len(pitch)>1:
            dp=np.abs(np.diff(pitch))
            s["pitch_step_abs_p50_p95_p99"]=[float(x) for x in np.percentile(dp,[50,95,99])]
            s["pitch_abs_gt70_pct"]=float(np.mean(np.abs(pitch)>70)*100)
            s["pitch_abs_gt80_pct"]=float(np.mean(np.abs(pitch)>80)*100)
        if len(mag):
            s["near_1g_pct"]=float(np.mean(np.abs(mag-1.0)<0.1)*100)
        if len(packet)>1:
            pd=np.diff(packet)
            s["packet_gap_gt1"]=int(np.sum(pd>1))
            s["packet_backwards"]=int(np.sum(pd<=0))
        # low-motion 0.5s blocks based on mag close to 1 and local angular std
        if len(yaw)>60 and len(mag)==len(yaw):
            wins=[]
            for i in range(0,len(yaw)-50,25):
                yy=np.rad2deg(np.unwrap(np.deg2rad(yaw[i:i+50])))
                pp=pitch[i:i+50] if len(pitch)==len(yaw) else np.array([])
                mm=mag[i:i+50]
                if len(pp)==50 and np.mean(np.abs(mm-1))<0.08 and np.std(yy)<3 and np.std(pp)<3:
                    yc=((np.mean(yy)+180)%360)-180
                    wins.append((yc,float(np.mean(pp)),float(np.std(yy)),float(np.std(pp))))
            s["stationary_windows"]=len(wins)
            if wins:
                ycs=np.array([x[0] for x in wins]); pcs=np.array([x[1] for x in wins])
                s["stationary_yaw_p05_p50_p95"]=[float(x) for x in np.percentile(ycs,[5,50,95])]
                s["stationary_pitch_p05_p50_p95"]=[float(x) for x in np.percentile(pcs,[5,50,95])]
        out[h]=s
    return out

report={}
for name,p in files:
    if p.exists():
        report[name]=summarize(p)
        report[name]["path"]=str(p)

out=ROOT/"HoloGrip_SpatialLab_0918"/"historical_raw_audit.json"
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
for name,d in report.items():
    print("\n===",name,"===")
    print(d["path"],"rows",d["rows"])
    for h in ("L","R"):
        if h not in d: continue
        s=d[h]
        print(h,"n",s["n"],"dur",round(s.get("duration_s",0),2),"hz",round(s.get("row_hz",0),2))
        print("  yaw", [round(x,2) for x in s.get("yaw_p01_p50_p99",[])],
              "pitch",[round(x,2) for x in s.get("pitch_p01_p50_p99",[])],
              "roll",[round(x,2) for x in s.get("roll_p01_p50_p99",[])])
        print("  step yaw",[round(x,3) for x in s.get("yaw_step_abs_p50_p95_p99",[])],
              "pitch",[round(x,3) for x in s.get("pitch_step_abs_p50_p95_p99",[])],
              "wrap",s.get("raw_wrap_crossings"),
              "pitch>70",round(s.get("pitch_abs_gt70_pct",0),2),"pitch>80",round(s.get("pitch_abs_gt80_pct",0),2))
        print("  pkt_gap",s.get("packet_gap_gt1"),"back",s.get("packet_backwards"),"near1g",round(s.get("near_1g_pct",0),1),
              "stationary",s.get("stationary_windows"))
        if s.get("stationary_windows"):
            print("  stationary yaw", [round(x,1) for x in s["stationary_yaw_p05_p50_p95"]],
                  "pitch",[round(x,1) for x in s["stationary_pitch_p05_p50_p95"]])
