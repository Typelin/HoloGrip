from pathlib import Path
import sys,time,threading,math
from collections import Counter
import serial

LAB=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918")
PROD=LAB/"HoloGrip_Production_vNext_0918"
sys.path.insert(0,str(PROD))
from production_core import FrameValidator

PORTS=["COM4","COM5"]
DUR=15.0
stats={p:{
    "lines":0,"parsed":0,"hand":Counter(),"invalid":0,"reasons":Counter(),
    "ids":[],"sensor_ms":[],"max_g":0.0,"cur_bad":0,"max_bad":0,
    "cur_good":0,"max_good":0,
} for p in PORTS}
lock=threading.Lock()

def parse(s):
    f=s.strip().split(",")
    if len(f)<10 or f[0]!="D" or f[1] not in ("L","R"): return None
    try:
        return dict(hand=f[1],ax=float(f[2]),ay=float(f[3]),az=float(f[4]),
                    yaw=float(f[5]),pitch=float(f[6]),roll=float(f[7]),
                    packet_id=int(f[8]),sensor_ms=int(f[9]))
    except: return None

def worker(port):
    st=stats[port]; v=FrameValidator()
    s=serial.Serial(port,460800,timeout=.08)
    try:s.reset_input_buffer()
    except:pass
    end=time.monotonic()+DUR
    while time.monotonic()<end:
        b=s.readline()
        if not b:continue
        st["lines"]+=1
        p=parse(b.decode("utf-8","ignore"))
        if p is None:continue
        st["parsed"]+=1;st["hand"][p["hand"]]+=1
        st["ids"].append(p["packet_id"]);st["sensor_ms"].append(p["sensor_ms"])
        g=math.sqrt(p["ax"]**2+p["ay"]**2+p["az"]**2)
        st["max_g"]=max(st["max_g"],g)
        ok,reason=v.check(p)
        if ok:
            st["cur_good"]+=1;st["max_good"]=max(st["max_good"],st["cur_good"])
            st["cur_bad"]=0
        else:
            st["invalid"]+=1;st["reasons"][reason]+=1
            st["cur_bad"]+=1;st["max_bad"]=max(st["max_bad"],st["cur_bad"])
            st["cur_good"]=0
    s.close()

ts=[threading.Thread(target=worker,args=(p,)) for p in PORTS]
start=time.monotonic()
for t in ts:t.start()
for t in ts:t.join()
elapsed=time.monotonic()-start

for p in PORTS:
    st=stats[p]
    gaps=0;dups=0;back=0;max_id_gap=0
    for a,b in zip(st["ids"],st["ids"][1:]):
        d=b-a
        if d==0:dups+=1
        elif d<0:back+=1
        elif d>1:gaps+=d-1;max_id_gap=max(max_id_gap,d-1)
    sensor_gaps=[]
    for a,b in zip(st["sensor_ms"],st["sensor_ms"][1:]):
        d=b-a
        if d>15:sensor_gaps.append(d)
    hz=st["parsed"]/elapsed
    print(p,{
        "elapsed_s":round(elapsed,2),
        "parsed":st["parsed"],"hz":round(hz,2),"hand":dict(st["hand"]),
        "invalid":st["invalid"],"invalid_pct":round(st["invalid"]/max(1,st["parsed"])*100,3),
        "reasons":dict(st["reasons"]),
        "max_bad_run_packets":st["max_bad"],
        "max_good_run_packets":st["max_good"],
        "packet_gap_missing":gaps,"packet_dup":dups,"packet_back":back,"max_id_gap":max_id_gap,
        "sensor_gap_count_gt15ms":len(sensor_gaps),"sensor_gap_max_ms":max(sensor_gaps) if sensor_gaps else 0,
        "max_g":round(st["max_g"],3),
    })
