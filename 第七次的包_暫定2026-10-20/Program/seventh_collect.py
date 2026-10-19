"""Raw-only collector; no detector threshold or model can remove CSV rows."""
import argparse
from collections import deque
import json
import time
from tkinter import messagebox

import numpy as np
from song_collection_ui import DesignedSongCollectionApp
from song_collection_server import parse_sensor_line
from production_core import FrameValidator


class Collection(DesignedSongCollectionApp):
    def __init__(self):
        self.recent={h:deque(maxlen=160) for h in 'LR'}
        self.observed_ports={}
        self.zero_ports=None
        self.zero_quality=None
        super().__init__(transport='serial')
        self.raw_only=True
        self.mode_var.set('raw_100hz');self.mode_segment.set('原始 100 Hz');self.mode_segment.configure(state='disabled')
        self._update_mode_description()
        self.title('HoloGrip｜10/20 第七次｜完整Raw 100Hz收集')
        self.file_var.set(f'CSV儲存：{self.output_dir.resolve()}')

    def _connect_serial_ports(self):
        self.observed_ports={};self.zero_ports=None
        for a in self.recent.values():a.clear()
        super()._connect_serial_ports()

    def _serial_loop(self,connection):
        while not self.stop_event.is_set() and getattr(connection,'is_open',False):
            try:line=connection.readline()
            except Exception:break
            if not line:continue
            packet=parse_sensor_line(line.decode('ascii',errors='replace'))
            if packet:
                old=self.observed_ports.get(connection.port)
                self.observed_ports[connection.port]=packet.hand if old in (None,packet.hand) else 'INVALID'
                self._process_packet(packet)

    def _process_packet(self,packet):
        super()._process_packet(packet)
        with self.hand_locks[packet.hand]:self.recent[packet.hand].append(packet)

    def ready_packets(self):
        current={c.port for c in self.serial_connections if getattr(c,'is_open',False)}
        mapping={p:h for p,h in self.observed_ports.items() if p in current}
        if len(current)!=2 or len(mapping)!=2 or set(mapping.values())!={'L','R'}:
            raise ValueError('需要兩個不同COM，分別穩定送出L與R')
        now=time.monotonic();result={}
        for h in 'LR':
            with self.hand_locks[h]:rows=[p for p in self.recent[h] if now-p.received_monotonic<=.6]
            if len(rows)<40 or now-rows[-1].received_monotonic>.2:raise ValueError(f'{h}資料不足或已斷線')
            dt=rows[-1].received_monotonic-rows[0].received_monotonic
            hz=(len(rows)-1)/dt if dt else 0
            if not 80<=hz<=120:raise ValueError(f'{h}為{hz:.1f}Hz，尚未穩定')
            validator=FrameValidator()
            for p in rows:
                ok,reason=validator.check(dict(ax=p.ax,ay=p.ay,az=p.az,yaw=p.yaw,pitch=p.pitch,roll=p.roll))
                if not ok:raise ValueError(f'{h}感測資料異常：{reason}')
            result[h]=rows
        return mapping,result

    def _calibrate_both(self):
        try:
            if getattr(self,'recorder',None) and self.recorder.active:raise ValueError('先停止並保存，不能在錄製中歸零')
            mapping,rows=self.ready_packets();now=time.monotonic();quality={};cal={}
            for h,a in rows.items():
                a=[p for p in a if now-p.received_monotonic<=.5]
                if len(a)<35:raise ValueError('手掌貼鼓面，靜止至少0.5秒')
                v=np.asarray([[p.ax,p.ay,p.az,p.yaw,p.pitch] for p in a])
                mag=np.linalg.norm(v[:,:3],axis=1);yaw=np.degrees(np.unwrap(np.radians(v[:,3])))
                if mag.std()>.15 or not .65<mag.mean()<1.35 or yaw.std()>3 or v[:,4].std()>3:raise ValueError(f'{h}尚未靜止，請手掌貼鼓面')
                cal[h]=(float(yaw.mean()%360),float(v[:,4].mean()))
                quality[h]=dict(samples=len(a),magnitude_std=float(mag.std()),yaw_std=float(yaw.std()),pitch_std=float(v[:,4].std()))
            for h,(yaw,pitch) in cal.items():
                with self.hand_locks[h]:self.detectors[h].yaw_offset=yaw;self.detectors[h].pitch_offset=pitch
            self.zero_ports=mapping;self.zero_quality=quality
            self._monitor_event('zero','',json.dumps(dict(calibration=cal,quality=quality),ensure_ascii=False))
            self._append_log('雙手鼓面歸零完成；拿起鼓棒，再開始歌曲收集。')
        except ValueError as exc:messagebox.showwarning('尚未就緒',str(exc))

    def _start_recording(self):
        try:
            mapping,_=self.ready_packets()
            if self.zero_ports!=mapping:raise ValueError('請先完成雙手鼓面歸零')
        except ValueError as exc:messagebox.showwarning('尚未就緒',str(exc));return
        super()._start_recording()

    def _refresh_ui(self):
        super()._refresh_ui()
        if getattr(self,'raw_only',False):
            for h in 'LR':self.hand_count_vars[h].set(f'原始封包 {self.packet_counts[h]:,}；不跑鼓種模型')

    def _stop_recording(self):
        recorder=getattr(self,'recorder',None)
        super()._stop_recording()
        if recorder is not None and recorder.path and not recorder.active:
            report=dict(schema='hologrip-seventh-raw-v1',raw_path=str(recorder.path),
                raw_written=recorder.written,raw_enqueued=recorder.enqueued,raw_dropped=recorder.dropped,
                saved_ok=recorder.saved_ok,write_error=recorder.write_error,written_by_hand=recorder.written_by_hand,
                zero_quality=self.zero_quality,ports=self.observed_ports,
                calibration={h:[d.yaw_offset,d.pitch_offset] for h,d in self.detectors.items()},
                mode='raw_100hz',model_loaded=False,hit_filter_applied_to_csv=False)
            recorder.path.with_suffix('.manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    app=Collection()
    if args.smoke:app.withdraw();app.update_idletasks();app.after(150,app._on_close)
    app.mainloop()


if __name__=='__main__':main()
