from pathlib import Path
import json, math
import numpy as np
from scipy.spatial.transform import Rotation as Rot

S=Path(r"C:\Users\Typelin_Station\Desktop\HoloGrip\第三次的包\Data\電源循環姿態測試\20260918_235201")
d=json.loads((S/"summary.json").read_text(encoding="utf-8"))
st=d["stages"]

def R(name):
    return Rot.from_euler("ZYX",st[name]["mean_rotation_euler_ZYX_deg"],degrees=True)
def g(name):
    x=np.array(st[name]["gravity_unit_mean"],float); return x/np.linalg.norm(x)
def angle(a,b):
    return math.degrees(math.acos(np.clip(float(np.dot(a,b)),-1,1)))
def gd(a,b):
    return math.degrees((R(a).inv()*R(b)).magnitude())

pairs=[("A0","A_reboot"),("B_before","B_after"),("A0","A_after_Bboot")]
for a,b in pairs:
    full=gd(a,b); tilt=angle(g(a),g(b))
    residual_lb=max(0,full-tilt)
    print(a,b,"full",round(full,3),"gravity",round(tilt,3),"full-minus-gravity",round(residual_lb,3),
          "eulerA",np.round(st[a]["mean_rotation_euler_ZYX_deg"],2),
          "eulerB",np.round(st[b]["mean_rotation_euler_ZYX_deg"],2))

# derive quaternion swing-twist of relative rotation about gravity axis of first pose;
# not causal proof, but quantifies rotation component not explained by shortest gravity alignment.
def align_vec(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b)
    v=np.cross(a,b); c=np.dot(a,b)
    if np.linalg.norm(v)<1e-9:
        if c>0:return Rot.identity()
        # 180 deg choose orthogonal
        axis=np.cross(a,[1,0,0])
        if np.linalg.norm(axis)<1e-6: axis=np.cross(a,[0,1,0])
        axis/=np.linalg.norm(axis)
        return Rot.from_rotvec(axis*np.pi)
    s=np.linalg.norm(v); axis=v/s
    return Rot.from_rotvec(axis*math.atan2(s,c))

print("\nResidual after minimal gravity alignment (diagnostic only):")
for a,b in pairs:
    rel=R(a).inv()*R(b)
    # gravity is body vector; minimal physical body-frame rotation taking ga->gb
    tiltrot=align_vec(g(a),g(b))
    resid=tiltrot.inv()*rel
    print(a,b,"tiltrot",round(math.degrees(tiltrot.magnitude()),3),"resid",round(math.degrees(resid.magnitude()),3),
          "resid_euler",np.round(resid.as_euler("ZYX",degrees=True),2))
