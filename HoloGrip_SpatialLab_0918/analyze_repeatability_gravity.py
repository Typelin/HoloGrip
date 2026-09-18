from pathlib import Path
import csv,json,math
import numpy as np

S=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\HoloGrip_SpatialLab_0918\repeatability_sessions\20260918_143857")
with (S/"raw.csv").open(encoding="utf-8",newline="") as f:
    rows=list(csv.DictReader(f))
res=json.loads((S/"result.json").read_text(encoding="utf-8"))

# Infer test start from t range: baseline A recorded 3.5..8.5 from stage start; raw t may have offset.
# Find first t and use same windows relative to first packet (connection already settled before stage start, tiny offset).
tmin=min(float(r["t"]) for r in rows)
A=(tmin+3.5,tmin+8.5)
B=(tmin+25.5,tmin+32.5)

def Rzyx(y,p,r):
    y,p,r=np.deg2rad([y,p,r])
    Rz=np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    Ry=np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rx=np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    return Rz@Ry@Rx

for h in ("L","R"):
    print("\n===",h,"===")
    arr={}
    for name,w in [("A",A),("B",B)]:
        rr=[r for r in rows if r["hand"]==h and w[0]<=float(r["t"])<=w[1]]
        acc=np.array([[float(r["ax"]),float(r["ay"]),float(r["az"])] for r in rr])
        m=acc.mean(axis=0); n=m/np.linalg.norm(m)
        arr[name]=(rr,m,n)
        print(name,"n",len(rr),"mean_acc",np.round(m,4),"norm",round(float(np.linalg.norm(m)),4),"unit",np.round(n,4))
    dot=float(np.clip(np.dot(arr["A"][2],arr["B"][2]),-1,1))
    angle=math.degrees(math.acos(dot))
    print("gravity_vector_angle_A_B_deg",round(angle,3))
    print("gravity_vector_delta",np.round(arr["B"][1]-arr["A"][1],4))
    sa=res["baseline_A"][h]; sb=res["baseline_B"][h]
    RA=Rzyx(sa["yaw"],sa["pitch"],sa["roll"]); RB=Rzyx(sb["yaw"],sb["pitch"],sb["roll"])
    D=RA.T@RB
    rotang=math.degrees(math.acos(np.clip((np.trace(D)-1)/2,-1,1)))
    print("ZYX_orientation_delta_deg_approx",round(rotang,3))
    print("Euler A",round(sa["yaw"],2),round(sa["pitch"],2),round(sa["roll"],2))
    print("Euler B",round(sb["yaw"],2),round(sb["pitch"],2),round(sb["roll"],2))
