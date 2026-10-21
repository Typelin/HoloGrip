"""Replay any compatible Raw CSV through the exact field Engine, without MIDI."""
import argparse
import csv
import json
from pathlib import Path

from threadpoolctl import threadpool_limits
import seventh_core as core


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();target=Path(args.output)
    if target.exists():raise ValueError('Refuse overwrite')
    bundle,config=core.load_components();engine=core.Engine(bundle,config)
    hands=core.trajectory.read_raw(args.raw)
    source=[]
    with Path(args.raw).open(encoding='utf-8-sig',newline='') as f:source=list(csv.DictReader(f))
    output=[]
    with threadpool_limits(limits=1):
        for _,h,s in sorted((s['row_index'],h,s) for h,a in hands.items() for s in a):
            row=source[s['row_index']]
            frame=dict(ax=s['ax'],ay=s['ay'],az=s['az'],yaw=float(row['yaw_deg']),pitch=float(row['pitch_deg']),roll=s['roll'])
            pred,_=engine.push(h,s,frame);output.extend(pred)
    target.parent.mkdir(parents=True,exist_ok=True)
    core.detector.rt.write_csv(target,output)
    manifest=dict(raw_sha256=core.detector.rt.sha256(args.raw),output_sha256=core.detector.rt.sha256(target),
        model_sha256=core.detector.rt.sha256(core.MODEL),detector_config_sha256=core.detector.rt.sha256(core.CONFIG),
        classifier_contract=core.runtime.CONTRACT,direction_modes=engine.runtime.hand_mode,
        zero_yaw=engine.runtime.zero_yaw,zero_pitch=engine.runtime.zero_pitch,
        rows=len(source),predictions=len(output),invalid=engine.invalid,
        midi_read_during_inference=False,uses_exact_field_engine=True,detector=config)
    target.with_suffix('.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
