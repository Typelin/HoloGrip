"""Frozen heading-domain adapter using only the already-recorded drumhead zero.

Keep the validated world-frame specialist near headings seen during training.
For new headings, use the classifier trained on heading-relative trajectories.
One external detector per hand; no class remap, labels, sequence or live fit.
"""
from collections import deque

import world_stream_runtime_1019 as legacy
import absolute_world_motion_runtime_1019 as world

CONTRACT='hologrip_recorded_zero_heading_adapter_v1'


def heading_mode(hand,zero_yaw,profile):
    distance=min(abs((zero_yaw-ref+180)%360-180) for ref in profile['reference_zero_yaw_deg'][hand])
    return ('trained_world' if distance<=profile['world_heading_radius_deg'] else 'drumhead_relative'),distance


class StreamEngine:
    def __init__(self,bundle,relative_bundle,profile):
        self.bundle=bundle;self.relative_bundle=relative_bundle;self.profile=profile
        self.history={h:deque(maxlen=1024) for h in 'LR'}
        self.hit={h:world.base.detector.HitStream(h,bundle['detector_config']) for h in 'LR'}
        self.hand_mode={h:None for h in 'LR'}
        self.zero_yaw={h:None for h in 'LR'}
        self.heading_distance={h:None for h in 'LR'}
        self.count=0

    def push(self,hand,sample):
        if self.hand_mode[hand] is None:
            zero=(sample['absolute_yaw']-sample['yaw'])%360
            self.hand_mode[hand],self.heading_distance[hand]=heading_mode(hand,zero,self.profile)
            self.zero_yaw[hand]=zero
        self.history[hand].append(sample)
        peaks=self.hit[hand].push(sample)
        if not peaks:return []
        rows=list(self.history[hand]);relative=self.hand_mode[hand]=='drumhead_relative'
        bundle=self.relative_bundle if relative else self.bundle
        # This is a coordinate conversion of an observed window, never a Raw edit.
        used=[dict(r,absolute_yaw=r['yaw']) for r in rows] if relative else rows
        prepared=world.base.legacy.prepare({hand:used});first=rows[0]['index'];output=[]
        for peak in peaks:
            local={**peak,'peak_index':int(peak['peak_index'])-first,'available_index':len(rows)-1}
            vector=world.feature(prepared,local,bundle['kind'])
            if vector is None:continue
            self.count+=1
            stamped={**peak,'prediction_id':self.count,'direction_mode':self.hand_mode[hand],
                'zero_yaw_deg':self.zero_yaw[hand],'heading_distance_deg':self.heading_distance[hand]}
            output.extend(legacy.classify(bundle,[stamped],{bundle['kind']:[vector]}))
        return output
