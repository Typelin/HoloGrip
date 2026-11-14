"""Raw-only past-trajectory MLP. No MIDI or annotation imports.

The detector is unchanged. Features see -240..+80 ms and never exceed the
packet index available at emission. Interpolation is a model representation,
not modification of Raw timestamps or MIDI alignment. Angular differences
are estimates from Euler samples, not measured gyroscope channels/position.
"""
import argparse
import csv
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import fourth_song03_raw_runtime_1008 as rt

CONTRACT='hologrip_past_trajectory_456_v1'
GRID=np.arange(-240.,81.,10.)


def read_raw(path):
    hands=rt.read_samples(path)
    counts={'L':0,'R':0}
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            h=row['hand'];s=hands[h][counts[h]];counts[h]+=1
            s['absolute_pitch']=float(row['pitch_deg'])
            s['sensor_t']=float(row['sensor_time_ms'])
    return hands


def prepare(hands):
    return {h:dict(rows=a,times=np.asarray([s['t'] for s in a])) for h,a in hands.items()}


def feature(prepared,p,variant):
    h=p['hand'];a=prepared[h]['rows'];ts=prepared[h]['times'];c=p['peak_ms']
    lo=int(np.searchsorted(ts,c-250,side='left'))
    hi=min(int(np.searchsorted(ts,c+80,side='right')),int(p['available_index'])+1)
    rows=a[lo:hi]
    if len(rows)<6:return None
    physical=variant!='calibrated_shape'
    use=[{**s,'pitch':s['absolute_pitch'] if physical else s['pitch']} for s in rows]
    base=rt.feature(use,c,h)
    if base is None:return None
    t=np.asarray([s['t'] for s in rows])-c
    accel=np.asarray([[s['ax'],s['ay'],s['az']] for s in rows])
    euler=np.degrees(np.unwrap(np.radians([[s['yaw'],s['absolute_pitch'],s['roll']] for s in rows]),axis=0))
    center=np.asarray([np.interp(0.,t,euler[:,i]) for i in range(3)])
    angle=euler-center
    body=np.column_stack([np.interp(GRID,t,accel[:,i]) for i in range(3)])
    mag=max(1.,float(np.linalg.norm(accel,axis=1).max()))
    body/=mag
    d=np.column_stack([np.interp(GRID,t,angle[:,i]) for i in range(3)])
    shape=np.column_stack([body,np.sin(np.radians(d[:,0])),np.cos(np.radians(d[:,0])),d[:,1]/90.,d[:,2]/90.])
    if variant=='gravity_world_shape':
        yaw,pitch,roll=np.radians(euler).T
        ax,ay,az=accel.T
        y=np.cos(roll)*ay-np.sin(roll)*az;z=np.sin(roll)*ay+np.cos(roll)*az
        x=np.cos(pitch)*ax+np.sin(pitch)*z;z=-np.sin(pitch)*ax+np.cos(pitch)*z
        world=np.column_stack([np.cos(yaw)*x-np.sin(yaw)*y,np.sin(yaw)*x+np.cos(yaw)*y,z-1.])
        world=np.column_stack([np.interp(GRID,t,world[:,i]) for i in range(3)])/mag
        shape=np.column_stack([shape,world])
    return np.r_[base,shape.ravel()].tolist()


def predict(bundle,hands,candidates):
    prep=prepare(hands);rows=[];vectors=[]
    for p in candidates:
        f=feature(prep,p,bundle['variant'])
        if f is not None:rows.append(p);vectors.append(f)
    if not rows:return []
    model=bundle['model'];probs=model.predict_proba(np.asarray(vectors))
    return [dict(**{k:v for k,v in p.items() if k!='feature'},drum=str(model.classes_[v.argmax()]),model_score=float(v.max())) for p,v in zip(rows,probs)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['raw','model','output']:ap.add_argument('--'+key,required=True)
    args=ap.parse_args();out=Path(args.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    b=joblib.load(args.model);assert b['runtime_contract']==CONTRACT
    hands=read_raw(args.raw);c,diag=rt.detect(hands,b['detector'])
    with threadpool_limits(limits=1):p=predict(b,hands,c)
    rt.write_csv(out,p)
    m=dict(raw_sha256=rt.sha256(args.raw),model_sha256=rt.sha256(args.model),output_sha256=rt.sha256(out),
           runtime_sha256=rt.sha256(__file__),predictions=len(p),detected=len(c),raw_rows=sum(map(len,hands.values())),
           input_dimensions=b['input_dimensions'],variant=b['variant'],feature_ms=[-240,80],
           uses_recorded_zero_only=True,uses_absolute_gravity_pitch=b['variant']!='calibrated_shape',
           midi_or_labels_read_during_inference=False,answer_derived_anchor=False,
           endpoint_padding='only already-observed endpoint values; no future packets',diagnostics=diag)
    out.with_suffix('.manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(m,ensure_ascii=False))


if __name__=='__main__':main()
