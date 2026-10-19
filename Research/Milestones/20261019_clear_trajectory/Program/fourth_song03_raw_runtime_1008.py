"""Raw CSV only: causal external peak detector followed by drum-only MLP.

No MIDI or labels are read here. The CSV's recorded calibration is retained.
The centered feature context waits for +80 ms; timestamp and decision time
are different fields. Missing tail context is never fabricated.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import deque
from pathlib import Path
import joblib
import numpy as np

CONTRACT = 'fourth_song03_external_peak_zone_v1'
FEATURES = ['max_mag','mean_mag','std_mag','energy','max_ax','max_ay','max_az',
            'jerk_max','mean_yaw','mean_pitch','mean_roll','std_yaw','std_pitch',
            'center_yaw','center_pitch','swing_depth','yaw_range','hand_R']
DEFAULT_DETECTOR = dict(threshold_g=2.3, refractory_ms=120.0, half_ms=80.0)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def read_samples(path):
    hands = {'L':[], 'R':[]}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for i,r in enumerate(csv.DictReader(f)):
            h=r['hand']
            v=[float(r[k]) for k in ('song_time_ms','ax_g','ay_g','az_g','cal_yaw_deg','cal_pitch_deg','roll_deg')]
            if h not in hands or not np.isfinite(v).all():
                raise ValueError(f'Invalid raw row {i+1}')
            if hands[h] and v[0]<hands[h][-1]['t']:
                raise ValueError('Backward raw clock')
            hands[h].append(dict(t=v[0], ax=v[1],ay=v[2],az=v[3],yaw=v[4],pitch=v[5],roll=v[6],
                                 mag=float(np.linalg.norm(v[1:4])), index=len(hands[h]), row_index=i))
    return hands

def feature(samples, center, hand, half=80):
    win=[s for s in samples if center-half<=s['t']<=center+half]
    if len(win)<6:
        return None
    m=np.asarray([s['mag'] for s in win]); a=np.asarray([[s['ax'],s['ay'],s['az']] for s in win])
    e=np.asarray([[s['yaw'],s['pitch'],s['roll']] for s in win])
    e=np.degrees(np.unwrap(np.radians(e),axis=0))
    k=int(np.argmin([abs(s['t']-center) for s in win]))
    return np.asarray([m.max(),m.mean(),m.std(),np.abs(m-1).sum(),*np.abs(a).max(axis=0),
                       np.abs(np.diff(m)).max(),*e.mean(axis=0),*e[:,:2].std(axis=0),
                       *e[k,:2],np.ptp(e[:,1]),np.ptp(e[:,0]),float(hand=='R')])

class PeakStream:
    """Only raw input. A plateau is one candidate, not one per equal sample.

    An already-confirmed stronger peak within the unresolved 80-ms window
    replaces a weaker precursor. A peak confirmed after emission cannot
    retroactively replace it. After emission the hand is refractory 120 ms.
    """
    def __init__(self, hand, **settings):
        self.hand=hand
        self.config={**DEFAULT_DETECTOR,**settings}
        self.buf=deque(maxlen=512)
        self.pending=None
        self.last=-1e12
        self.suppressed=0
        self.replaced=0

    def push(self,s):
        self.buf.append(s)
        b=list(self.buf)
        if len(b)>=7:
            c=b[-4]; m=c['mag']; cfg=self.config
            # Strict on the left chooses the first packet of a flat top.
            is_peak=m>=cfg['threshold_g'] and m>max(x['mag'] for x in b[-7:-4]) and m>=max(x['mag'] for x in b[-3:])
            if is_peak:
                if c['t']-self.last<cfg['refractory_ms']:
                    self.suppressed+=1
                elif self.pending is None:
                    self.pending=c
                elif c['t']-self.pending['t']<cfg['half_ms']:
                    if m>self.pending['mag']:
                        self.pending=c; self.replaced+=1
                    else:
                        self.suppressed+=1
        if self.pending is None or s['t']<self.pending['t']+self.config['half_ms']:
            return []
        p=self.pending; self.pending=None; self.last=p['t']
        v=feature(b,p['t'],self.hand,self.config['half_ms'])
        if v is None:
            return []
        return [dict(hand=self.hand,peak_ms=p['t'],emitted_ms=s['t'],peak_g=p['mag'],
                     peak_index=p['index'],available_index=s['index'],feature=v.tolist(),
                     window_samples=sum(p['t']-80<=x['t']<=p['t']+80 for x in b))]

def detect(hands,settings=None):
    out=[]; diagnostics={}
    for h,samples in hands.items():
        stream=PeakStream(h,**(settings or {}))
        for s in samples:
            out.extend(stream.push(s))
        diagnostics[h]=dict(rows=len(samples),suppressed=stream.suppressed,replaced=stream.replaced,
                            incomplete_tail=int(stream.pending is not None))
    out.sort(key=lambda p:(p['peak_ms'],p['hand']))
    for i,p in enumerate(out):
        p['prediction_id']=i+1
    return out,diagnostics

def classify(model,candidates):
    if not candidates:
        return []
    scores=model.predict_proba(np.asarray([p['feature'] for p in candidates]))
    return [dict(**p,drum=str(model.classes_[int(np.argmax(v))]),model_score=float(np.max(v)))
            for p,v in zip(candidates,scores)]

def write_csv(path,rows):
    if not rows:
        raise ValueError('Refuse ambiguous empty CSV export')
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def predict_raw(raw_path,model_path,output):
    bundle=joblib.load(model_path)
    if bundle['runtime_contract']!=CONTRACT or bundle['feature_names']!=FEATURES:
        raise ValueError('Wrong model contract')
    hands=read_samples(raw_path)
    cand,diagnostics=detect(hands,bundle['detector'])
    pred=classify(bundle['model'],cand)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    write_csv(output,[{k:v for k,v in p.items() if k!='feature'} for p in pred])
    meta=dict(runtime_contract=CONTRACT,raw_sha256=sha256(raw_path),model_sha256=sha256(model_path),
              output_sha256=sha256(output),prediction_count=len(pred),diagnostics=diagnostics,
              midi_or_labels_read_during_inference=False,calibration='recorded CSV calibration; no prefix re-zero',
              feature_window_ms=[-80,80],classes=list(bundle['model'].classes_),detector=bundle['detector'])
    output.with_suffix('.manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False),flush=True)
    return pred

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',required=True);parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();predict_raw(a.raw,a.model,a.output)
