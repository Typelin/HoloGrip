from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy.spatial.transform import Rotation as Rot

FEATURE_NAMES = [
    "max_mag","mean_mag","std_mag","energy","jerk_max",
    "max_local_ax","max_local_ay","max_local_az",
    "mean_local_ax","mean_local_ay","mean_local_az",
    "rot_angle_std","rot_angle_range",
    "center_r6_0","center_r6_1","center_r6_2","center_r6_3","center_r6_4","center_r6_5",
    "mean_r6_0","mean_r6_1","mean_r6_2","mean_r6_3","mean_r6_4","mean_r6_5",
    "std_r6_0","std_r6_1","std_r6_2","std_r6_3","std_r6_4","std_r6_5",
]

ZONE_NAMES = ["小鼓","高音 Tom","中音 Tom","落地 Tom","Hi-Hat","Crash","Ride"]

def circ_mean_deg(vals: Iterable[float]) -> float:
    a = np.asarray(list(vals), float)
    r = np.deg2rad(a)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))))

def compute_r0(eulers_zyx_deg: list[tuple[float,float,float]]) -> tuple[np.ndarray, tuple[float,float,float]]:
    if len(eulers_zyx_deg) < 20:
        raise ValueError(f"校正樣本不足: {len(eulers_zyx_deg)}")
    a=np.asarray(eulers_zyx_deg,float)
    y0=circ_mean_deg(a[:,0])
    p0=float(np.median(a[:,1]))
    r0=circ_mean_deg(a[:,2])
    R0=Rot.from_euler("ZYX",[y0,p0,r0],degrees=True).as_matrix()
    return R0,(y0,p0,r0)

def r6(M: np.ndarray) -> np.ndarray:
    return M[:, :2].reshape(-1)

def make_relative_sample(
    R0: np.ndarray,
    t_ms: float,
    ax: float, ay: float, az: float,
    yaw: float, pitch: float, roll: float,
) -> dict:
    Rc=Rot.from_euler("ZYX",[yaw,pitch,roll],degrees=True).as_matrix()
    Rrel=R0.T @ Rc
    a=np.asarray([ax,ay,az],float)
    alocal=R0.T @ Rc @ a
    mag=float(np.linalg.norm(a))
    rel_euler=Rot.from_matrix(Rrel).as_euler("ZYX",degrees=True)
    return {
        "t":float(t_ms),
        "mag":mag,
        "alocal":alocal,
        "Rrel":Rrel,
        "rel_yaw":float(rel_euler[0]),
        "rel_pitch":float(rel_euler[1]),
        "rel_roll":float(rel_euler[2]),
    }

def feature_from_samples(samples, center_ms: float, half_ms: float=80.0):
    w=[s for s in samples if center_ms-half_ms <= s["t"] <= center_ms+half_ms]
    if not w:
        return None
    mag=np.asarray([s["mag"] for s in w],float)
    A=np.vstack([s["alocal"] for s in w])
    mats=np.stack([s["Rrel"] for s in w])
    near=min(w,key=lambda s:abs(s["t"]-center_ms))
    r6s=np.stack([r6(M) for M in mats])
    center6=r6(near["Rrel"])
    mean6=np.mean(r6s,axis=0)
    std6=np.std(r6s,axis=0)
    angles=np.asarray([Rot.from_matrix(M).magnitude() for M in mats],float)
    return np.asarray([
        np.max(mag),np.mean(mag),np.std(mag),np.sum(np.abs(mag-1.0)),
        np.max(np.abs(np.diff(mag))) if len(mag)>1 else 0.0,
        *np.max(np.abs(A),axis=0).tolist(),
        *np.mean(A,axis=0).tolist(),
        float(np.std(angles)),float(np.max(angles)-np.min(angles)),
        *center6.tolist(),*mean6.tolist(),*std6.tolist(),
    ],float)
