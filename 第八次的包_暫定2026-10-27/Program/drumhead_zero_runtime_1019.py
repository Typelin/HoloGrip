"""Raw-only external hits -> drum classifier, using recorded zero only.

No MIDI, expected hit, snare identity, answer-derived reference, take-wide
statistics or initial calibration taps are accepted. Two-axis zero is the
recorded yaw/pitch operation, not a claim of full glove-mount invariance.
Both batch and packet streaming use no samples beyond the emission packet.
"""
import argparse
import json
from collections import deque
from pathlib import Path
import joblib
import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
import trajectory_runtime_456_1016 as legacy
import causal_hit_detector_1016 as detector
import causal_hit_queue_1016 as queued

CONTRACT='hologrip_drumhead_zero_ensemble_1019_v1'
GRID=np.arange(-120.,81.,10.)
POSE=np.arange(0,len(GRID),4)
KINDS=['cyclic_zero','two_axis_rotation']


def feature(prepared,p,kind):
    ts=prepared[p['hand']]['times'];a=prepared[p['hand']]['rows'];c=p['peak_ms']
    lo=int(np.searchsorted(ts,c-130.,side='left'))
    hi=min(int(np.searchsorted(ts,c+80.,side='right')),int(p['available_index'])+1)
    rows=a[lo:hi]
    if len(rows)<6:return None
    base=legacy.rt.feature(rows,c,p['hand'])
    if base is None:return None
    t=ts[lo:hi]-c
    accel=np.asarray([[s['ax'],s['ay'],s['az']] for s in rows])
    angles=np.degrees(np.unwrap(np.radians([[s['yaw'],s['absolute_pitch'],s['roll']] for s in rows]),axis=0))
    eg=np.column_stack([np.interp(GRID,t,angles[:,i]) for i in range(3)])
    center=np.asarray([np.interp(0.,t,angles[:,i]) for i in range(3)])
    body=np.column_stack([np.interp(GRID,t,accel[:,i]) for i in range(3)])
    scale=max(1.,float(np.linalg.norm(accel,axis=1).max()))
    # This offset is directly available in the current Raw/online sample.
    # It is never inferred from labelled drum hits or a future recording.
    zero_pitch=rows[-1]['absolute_pitch']-rows[-1]['pitch']
    rotation=Rotation.from_euler('ZYX',eg,degrees=True)
    rotation_center=Rotation.from_euler('ZYX',center,degrees=True)
    linear=rotation.apply(body)-np.array([0.,0.,1.])
    if kind=='cyclic_zero':
        cp=eg.copy();cp[:,1]-=zero_pitch
        pose=np.column_stack([np.sin(np.radians(cp[POSE])),np.cos(np.radians(cp[POSE]))]).ravel()
    elif kind=='two_axis_rotation':
        zero=Rotation.from_euler('Y',-zero_pitch,degrees=True)
        rotation=zero*rotation;rotation_center=zero*rotation_center
        linear=zero.apply(linear)
        pose=rotation[POSE].as_matrix().ravel()
    else:raise ValueError(kind)
    relative=(rotation*rotation_center.inv()).as_rotvec()/np.pi
    angular=[8,9,10,13,14]
    scalar=np.asarray([v for i,v in enumerate(base) if i not in angular])
    cyclic=np.column_stack([np.sin(np.radians(base[angular])),np.cos(np.radians(base[angular]))]).ravel()
    wave=np.column_stack([body/scale,linear/scale,relative]).ravel()
    return np.r_[scalar,cyclic,pose,wave].astype(np.float32)


def probabilities(bundle,X):
    mode=bundle['mode'];models=bundle['components']
    if mode=='mlp':return models['mlp'].predict_proba(models['scaler'].transform(X))
    if mode=='gbdt':return models['gbdt'].predict_proba(X)
    return .5*(models['mlp'].predict_proba(models['scaler'].transform(X))+models['gbdt'].predict_proba(X))


def classify(bundle,rows,vectors):
    if not rows:return []
    probs=probabilities(bundle,np.asarray(vectors,dtype=np.float32))
    return [dict(**{k:v for k,v in p.items() if k!='feature'},
        drum=str(bundle['classes'][q.argmax()]),model_score=float(q.max()),
        model_margin=float(np.sort(q)[-1]-np.sort(q)[-2])) for p,q in zip(rows,probs)]


def predict(bundle,hands,peaks):
    prep=legacy.prepare(hands);rows=[];vectors=[]
    for p in peaks:
        x=feature(prep,p,bundle['kind'])
        if x is not None:rows.append(p);vectors.append(x)
    return classify(bundle,rows,vectors)


def detect(hands,profile):
    return (queued if profile['implementation']=='queue' else detector).detect(hands,profile['config'])


class StreamEngine:
    """An actual packet interface; needs no teacher/protocol/MIDI argument."""
    def __init__(self,bundle):
        self.bundle=bundle;self.history={h:deque(maxlen=512) for h in 'LR'};self.count=0
        p=bundle['detector_profile']
        self.joint=queued.JointStream(p['config']) if p['implementation']=='queue' else None
        self.single=({h:detector.HitStream(h,p['config']) for h in 'LR'} if self.joint is None else None)

    def push(self,hand,sample):
        self.history[hand].append(sample)
        peaks=self.joint.push(hand,sample) if self.joint else self.single[hand].push(sample)
        out=[]
        for p in peaks:
            prepared=legacy.prepare({hand:list(self.history[hand])})
            local={**p,'available_index':len(self.history[hand])-1}
            x=feature(prepared,local,self.bundle['kind'])
            if x is None:continue
            self.count+=1;p={**p,'prediction_id':self.count}
            out.extend(classify(self.bundle,[p],[x]))
        return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['raw','model','output']:ap.add_argument('--'+key,required=True)
    args=ap.parse_args();out=Path(args.output);assert not out.exists()
    b=joblib.load(args.model);assert b['runtime_contract']==CONTRACT
    hands=legacy.read_raw(args.raw)
    hits,diag=detect(hands,b['detector_profile'])
    with threadpool_limits(limits=1):pred=predict(b,hands,hits)
    out.parent.mkdir(parents=True,exist_ok=True);legacy.rt.write_csv(out,pred)
    meta=dict(raw=str(Path(args.raw).resolve()),raw_sha256=legacy.rt.sha256(args.raw),
        model_sha256=legacy.rt.sha256(args.model),runtime_sha256=legacy.rt.sha256(__file__),
        output_sha256=legacy.rt.sha256(out),rows=sum(map(len,hands.values())),detected=len(hits),predictions=len(pred),
        feature_ms=[-120,80],midi_or_labels_read_during_inference=False,
        answer_derived_anchor=False,additional_calibration_taps=False,recording_wide_statistics=False,
        external_hit_detection=True,learned_hit_background_class=False,abstention=False,diagnostics=diag)
    out.with_suffix('.manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False))


if __name__=='__main__':main()
