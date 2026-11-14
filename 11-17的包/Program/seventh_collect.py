"""Raw collector; no detector threshold or model can remove CSV rows.

Live judgement (collect_live_judge.LiveJudge) only shows which drum each hit is judged as and writes a
side file; the raw CSV, its queue and the packets are unchanged."""
import argparse
from collections import deque
import json
import os
import threading
import time
from tkinter import messagebox

import numpy as np
from song_collection_ui import DesignedSongCollectionApp
from song_collection_server import parse_sensor_line
from production_core import FrameValidator
import glove_udp_link as udp
import glove_serial_link
import collect_live_judge

SILENT_S=0.5   # a hand with no packet for this long during a take is a dropout (same rule as the live program)
UDP_MODE=os.environ.get('HOLOGRIP_TRANSPORT','').lower()=='udp'


class Collection(DesignedSongCollectionApp):
    def __init__(self):
        self.recent={h:deque(maxlen=160) for h in 'LR'}
        self.observed_ports={}
        self.zero_ports=None
        self.zero_quality=None
        self.last_arrival={h:None for h in 'LR'};self.silent={h:None for h in 'LR'};self.silent_periods=[];self.take_t0=None
        super().__init__(transport='serial')
        self.raw_only=True
        self.mode_var.set('raw_100hz');self.mode_segment.set('原始 100 Hz');self.mode_segment.configure(state='disabled')
        self._update_mode_description()
        self.title('HoloGrip｜11/17 的包｜完整Raw 100Hz收集')
        self.file_var.set(f'CSV儲存：{self.output_dir.resolve()}')
        self.judge=collect_live_judge.LiveJudge(log=lambda m:self.ui_queue.put(('log',m)))
        self.judge_summary=None

    def _refresh_serial_ports(self):
        super()._refresh_serial_ports()
        values=list(self.com_a_box.cget('values') or [])
        values=[v for v in values if not v.startswith('未找到')]+list(udp.PORT_NAMES)
        self.com_a_box.configure(values=values);self.com_b_box.configure(values=values)
        if UDP_MODE:self.com_a_var.set('UDP-L');self.com_b_var.set('UDP-R')

    def _connect_serial_ports(self):
        self.observed_ports={};self.zero_ports=None
        for a in self.recent.values():a.clear()
        selected=[self.com_a_var.get().strip(),self.com_b_var.get().strip()]
        if not any(p in udp.PORT_NAMES for p in selected):return self._connect_com(selected)
        if sorted(selected)!=['UDP-L','UDP-R']:
            messagebox.showwarning('無線埠','無線版請左右各選 UDP-L 與 UDP-R（不能和 COM 混用）');return
        self._close_serial_ports()
        try:
            for name in selected:
                c=udp.open_port(name);self.serial_connections.append(c)
                threading.Thread(target=self._serial_loop,args=(c,),daemon=True,name=f'udp-{name}').start()
        except OSError as exc:
            self._close_serial_ports();messagebox.showerror('無線接收失敗',f'UDP {udp.DATA_PORT} 無法使用：{exc}');return
        self._start_monitor(selected)
        self.status_var.set('無線已啟動，等待左右手套（手機熱點）送資料')
        self.serial_connect_button.configure(text='重新連接')
        self._append_log(f'無線接收已啟動：UDP {udp.DATA_PORT}，每秒廣播尋找手套')

    def _connect_com(self,selected):
        """Wired: same checks as before, but each COM reopens itself if the glove stops sending (glove_serial_link)."""
        selected=[p for p in selected if p and not p.startswith(('選擇','未找到'))]
        if not selected:messagebox.showwarning('尚未選擇','請先選擇至少一個 COM 埠。');return
        if len(set(selected))!=len(selected):messagebox.showwarning('COM 埠重複','左右手不能選同一個 COM 埠。');return
        self._close_serial_ports()
        try:
            for port in selected:
                c=glove_serial_link.ReconnectingSerial(port);self.serial_connections.append(c)
                threading.Thread(target=self._serial_loop,args=(c,),daemon=True,name=f'serial-{port}').start()
        except Exception as exc:
            self._close_serial_ports();messagebox.showerror('COM 連線失敗',str(exc));return
        self._start_monitor(selected)
        self.status_var.set('COM 已連接，等待左右手 D 封包')
        self.serial_connect_button.configure(text='重新連接 COM')
        self._append_log(f"COM 已連接：{', '.join(selected)}（460800 baud；手套停送 {glove_serial_link.SILENT_REOPEN_S:g} 秒會自動重開）")

    def _serial_loop(self,connection):
        wireless=hasattr(connection,'read_packet')
        while not self.stop_event.is_set() and getattr(connection,'is_open',False):
            try:
                if wireless:packet=connection.read_packet(0.2)
                else:
                    line=connection.readline()
                    packet=parse_sensor_line(line.decode('ascii',errors='replace')) if line else None
            except Exception:break
            if packet:
                old=self.observed_ports.get(connection.port)
                self.observed_ports[connection.port]=packet.hand if old in (None,packet.hand) else 'INVALID'
                self._process_packet(packet)

    def _process_packet(self,packet):
        super()._process_packet(packet)
        with self.hand_locks[packet.hand]:self.recent[packet.hand].append(packet)
        self.last_arrival[packet.hand]=time.monotonic()
        judge=getattr(self,'judge',None)
        if judge is not None:judge.feed(packet)

    def _watch_silence(self):
        """During a take: a hand silent > SILENT_S is logged, shown and written to the monitor and the manifest."""
        recorder=getattr(self,'recorder',None)
        if recorder is None or not recorder.active or self.take_t0 is None:return
        now=time.monotonic()
        for h,name in (('L','左'),('R','右')):
            last=self.last_arrival[h] or self.take_t0
            if self.silent[h] is None and now-last>SILENT_S:
                self.silent[h]=last
                msg=f'⚠ {name}手超過{SILENT_S}秒沒有資料（斷線？）；這段不能用，請停止並檢查手套'
                self.status_var.set(msg);self._append_log(msg)
                self._monitor_event('hand_silent','',f'{h}|take_s={last-self.take_t0:.2f}')
            elif self.silent[h] is not None and now-last<=SILENT_S:
                self.silent_periods.append(dict(hand=h,start_s=round(self.silent[h]-self.take_t0,3),end_s=round(last-self.take_t0,3)))
                self._monitor_event('hand_resume','',f'{h}|take_s={last-self.take_t0:.2f}')
                self._append_log(f'{name}手恢復資料（中間 {last-self.silent[h]:.2f} 秒缺資料）');self.silent[h]=None

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
        self.silent={h:None for h in 'LR'};self.silent_periods=[];self.take_t0=time.monotonic()
        super()._start_recording()
        recorder=getattr(self,'recorder',None)
        if recorder is not None and recorder.active:
            cal={h:(d.yaw_offset,d.pitch_offset) for h,d in self.detectors.items()}
            if not self.judge.start(recorder._start_monotonic,cal,recorder.path):
                self._append_log('即時判斷模型還沒載入好，這一段只收 Raw、不顯示判斷' if self.judge.load_error is None else '即時判斷未啟用；這一段只收 Raw')

    def _refresh_ui(self):
        super()._refresh_ui()
        self._watch_silence()
        if getattr(self,'raw_only',False):
            judge=getattr(self,'judge',None)
            for h in 'LR':
                self.hand_count_vars[h].set(f'原始封包 {self.packet_counts[h]:,}；Raw 不經模型過濾')
                text=judge.view(h) if judge is not None and judge.take is not None else None
                if text:self.hand_latest_vars[h].set(text)

    def _stop_recording(self):
        recorder=getattr(self,'recorder',None)
        if recorder is not None and recorder.active and self.take_t0 is not None:
            now=time.monotonic()
            for h in 'LR':
                last=self.last_arrival[h] or self.take_t0
                if self.silent[h] is not None or now-last>SILENT_S:
                    self.silent_periods.append(dict(hand=h,start_s=round((self.silent[h] or last)-self.take_t0,3),end_s=None))
                self.silent[h]=None
            self.take_t0=None
        super()._stop_recording()
        self.judge_summary=self.judge.stop() if getattr(self,'judge',None) is not None else None
        if self.judge_summary:
            j=self.judge_summary
            self._append_log(f"即時判斷：{j['predictions']} 下（左 {sum(j['by_hand']['L'].values())}、右 {sum(j['by_hand']['R'].values())}）"+(f"；⚠ 判斷佇列滿掉 {j['queue_dropped']} 筆（Raw 不受影響）" if j['queue_dropped'] else ''))
        if recorder is not None and recorder.path and not recorder.active:
            report=dict(schema='hologrip-seventh-raw-v1',raw_path=str(recorder.path),
                raw_written=recorder.written,raw_enqueued=recorder.enqueued,raw_dropped=recorder.dropped,
                saved_ok=recorder.saved_ok,write_error=recorder.write_error,written_by_hand=recorder.written_by_hand,
                zero_quality=self.zero_quality,ports=self.observed_ports,
                calibration={h:[d.yaw_offset,d.pitch_offset] for h,d in self.detectors.items()},
                mode='raw_100hz',model_loaded=bool(self.judge_summary),hit_filter_applied_to_csv=False,
                live_judge=self.judge_summary,
                hand_silent_periods=self.silent_periods,sensor_clean=not self.silent_periods,
                transport={c.port:(c.stats() if hasattr(c,'stats') else 'serial') for c in self.serial_connections})
            recorder.path.with_suffix('.manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    app=Collection()
    if args.smoke:app.withdraw();app.update_idletasks();app.after(150,app._on_close)
    app.mainloop()


if __name__=='__main__':main()
