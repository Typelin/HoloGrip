"""Serial -> bounded queue -> recording/inference worker for the native GUI.

Raw is saved before sensor validation. Browser rendering never owns COM or
recording. Testing begins immediately; no demonstration or adaptation stage.
"""
import argparse
import csv
from datetime import datetime
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import threading
import time
import traceback
from collections import deque

import numpy as np
import serial
from serial.tools import list_ports
from threadpoolctl import threadpool_limits

from song_collection_server import AsyncCsvRecorder,parse_sensor_line
from production_core import FrameValidator
import seventh_core as core

PACK=core.PACK
PRED_FIELDS=['prediction_id','hand','peak_ms','emitted_ms','peak_g','drum','model_score',
    'model_margin','accepted','guard_reason','peak_index','available_index','window_samples','processing_latency_ms','phase',
    'direction_mode','zero_yaw_deg','heading_distance_deg','hit_probability']


class Take:
    def __init__(self,parent,sid,bundle,config,calibration,ports,clock=time.monotonic):
        self.clock=clock
        self.folder=Path(parent)/(sid+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.folder.mkdir(parents=True,exist_ok=False)
        self.recorder=AsyncCsvRecorder(self.folder,lambda text:None)
        self.recorder.require_both_hands=True
        self.recorder.start('raw_100hz',sid)
        self.recorder._start_monotonic=self.clock()
        self.calibration=dict(calibration)
        self.ports=dict(ports)
        self.engine=core.Engine(bundle,config)
        self.indices={h:0 for h in 'LR'}
        self.last_ids={h:None for h in 'LR'}
        self.gaps={h:0 for h in 'LR'}
        self.resets={h:0 for h in 'LR'}
        self.phase='formal'
        self.formal_counts={h:0 for h in 'LR'}
        self.last_hit={h:None for h in 'LR'}
        self.last_decision={h:None for h in 'LR'}
        self.rejected_counts={h:0 for h in 'LR'}
        self.marker_count=0
        self.current_item=None
        self.started=datetime.now().astimezone().isoformat()
        self.pred_file=(self.folder/'predictions.csv').open('w',encoding='utf-8-sig',newline='')
        self.pred_writer=csv.DictWriter(self.pred_file,fieldnames=PRED_FIELDS)
        self.pred_writer.writeheader()
        self.marker_file=(self.folder/'markers.csv').open('w',encoding='utf-8-sig',newline='')
        self.marker_writer=csv.DictWriter(self.marker_file,fieldnames=['marker_index','t_ms','kind','expected_hand','expected_drum','comment'])
        self.marker_writer.writeheader()
        self.last_flush=self.clock()
        self.closed=False
        self.marker('session_start',comment='Frozen trajectory; direct test; no personal adaptation.')
        self.marker('formal_start',comment='Immediate model output. No demonstration, fitting or target hints.')

    def elapsed(self):return max(0.,(self.clock()-self.recorder._start_monotonic)*1000)

    def marker(self,kind,hand='',drum='',comment=''):
        self.marker_count+=1
        self.marker_writer.writerow(dict(marker_index=self.marker_count,t_ms=round(self.elapsed(),3),
            kind=kind,expected_hand=hand,expected_drum=drum,comment=comment))
        self.marker_file.flush()

    def push(self,packet):
        if packet.received_monotonic<self.recorder._start_monotonic:return [],''
        h=packet.hand
        t=max(0,int(round((packet.received_monotonic-self.recorder._start_monotonic)*1000)))
        sample=core.packet_sample(packet,t,self.calibration[h],self.indices[h])
        row=core.raw_row(packet,sample,sum(self.indices.values()))
        if not self.recorder.enqueue(row,packet.received_monotonic):raise RuntimeError('Raw寫入佇列滿或已關閉')
        self.indices[h]+=1
        prior=self.last_ids[h]
        if prior is not None and packet.packet_id>=0:
            if packet.packet_id>prior+1:self.gaps[h]+=packet.packet_id-prior-1
            elif packet.packet_id<=prior:self.resets[h]+=1
        self.last_ids[h]=packet.packet_id
        frame=dict(ax=packet.ax,ay=packet.ay,az=packet.az,yaw=packet.yaw,pitch=packet.pitch,roll=packet.roll)
        predictions,reason=self.engine.push(h,sample,frame)
        for p in predictions:
            p['phase']='formal'
            p['processing_latency_ms']=round(max(0.,self.clock()-packet.received_monotonic)*1000,3)
            self.pred_writer.writerow({k:p[k] for k in PRED_FIELDS})
            self.last_decision[h]=dict(p)
            if p['accepted']:
                self.formal_counts[h]+=1;self.last_hit[h]=dict(p)
            else:self.rejected_counts[h]+=1
        if predictions or self.clock()-self.last_flush>1:
            self.pred_file.flush();self.last_flush=self.clock()
        return predictions,reason

    def begin_item(self,hand,drum):
        if hand not in 'LR' or len(hand)!=1 or drum not in core.DRUMS:
            raise ValueError('請選有效手別與鼓種')
        self.end_item()
        self.current_item=(hand,drum)
        self.marker('item_start',hand,drum,'Record only; never supplied to the classifier.')

    def end_item(self):
        if self.current_item:
            self.marker('item_end',*self.current_item)
            self.current_item=None

    def snapshot(self):
        return dict(folder=str(self.folder),phase=self.phase,raw_enqueued=self.recorder.enqueued,
            raw_written=self.recorder.written,elapsed_ms=round(self.elapsed()),formal_counts=dict(self.formal_counts),
            last_hit=dict(self.last_hit),last_decision=dict(self.last_decision),rejected_counts=dict(self.rejected_counts),
            direction_mode=dict(self.engine.runtime.hand_mode))

    def stop(self,drops=0,errors=None):
        if self.closed:raise ValueError('本場已停止')
        errors=list(errors or [])
        try:
            self.end_item()
            self.marker('session_stop')
        except Exception as exc:errors.append('marker write: '+str(exc))
        path,enqueued,written,dropped=self.recorder.stop()
        for stream in [self.pred_file,self.marker_file]:
            try:stream.flush();stream.close()
            except Exception as exc:errors.append('metadata close: '+str(exc))
        self.closed=True
        def safe_sha(p):
            try:return core.detector.rt.sha256(p)
            except Exception as exc:errors.append('checksum: '+str(exc));return None
        result=dict(schema='hologrip-seventh-session-v1',started_at=self.started,
            stopped_at=datetime.now().astimezone().isoformat(),raw_path=str(path),raw_enqueued=enqueued,
            raw_written=written,raw_dropped=dropped,input_queue_dropped=drops,
            written_by_hand=self.recorder.written_by_hand,raw_write_error=self.recorder.write_error,
            packet_gaps=self.gaps,packet_resets_or_duplicates=self.resets,invalid_by_hand=self.engine.invalid,
            calibration=self.calibration,ports=self.ports,detector=self.engine.config,
            model_sha256=core.detector.rt.sha256(core.MODEL),config_sha256=core.detector.rt.sha256(core.CONFIG),
            output_guard_sha256=None,output_guard=self.engine.output_guard,
            classifier_contract=core.runtime.CONTRACT,detector_contract=core.detector.CONTRACT,
            input_dimensions=core.GENERATED_DIMENSIONS,selected_dimensions=core.SELECTED_DIMENSIONS,
            model_name=core.MODEL_NAME,training_events=self.engine.bundle['training_events'],
            venue_free=dict(contract=core.runtime.CONTRACT,compass_heading_used=False,
                pretests_in_training=False,per_hit_independent=True,future_context_ms=80,
                hand_mode=self.engine.runtime.hand_mode,zero_yaw=self.engine.runtime.zero_yaw,
                zero_pitch=self.engine.runtime.zero_pitch),
            training_scope='songs+four-drum+clear single-hand (round4/5/6 pretests excluded)',prediction_count=self.engine.count,
            formal_counts=self.formal_counts,rejected_counts=self.rejected_counts,rejected_predictions_preserved=True,
            demonstration_required=False,
            reference_targets_used_during_inference=False,personal_adaptation=False,neural_weights_modified=False,
            test_mode='direct_frozen_model_test',midi_read_during_inference=False,
            pending_tail={h:int(s.pending is not None) for h,s in self.engine.streams.items()},
            raw_sha256=safe_sha(path),predictions_sha256=safe_sha(self.folder/'predictions.csv'),
            markers_sha256=safe_sha(self.folder/'markers.csv'),errors=errors,
            sensor_clean=not any(self.engine.invalid.values()) and not any(self.gaps.values()) and not any(self.resets.values()),
            saved_ok=bool(self.recorder.saved_ok and drops==0 and not errors))
        (self.folder/'session_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result


class Service:
    def __init__(self,data_root=None,clock=time.monotonic,ports_provider=None,serial_factory=None):
        self.bundle,self.config=core.load_components()
        self.data_root=Path(data_root) if data_root else PACK/'Data'
        self.clock=clock
        self.ports_provider=ports_provider or (lambda:[p.device for p in list_ports.comports()])
        self.serial_factory=serial_factory or (lambda port:serial.Serial(port,460800,timeout=.2))
        self.tasks=queue.Queue(maxsize=50000);self.lock=threading.RLock()
        self.read_stop=threading.Event();self.connections=[];self.readers=[]
        self.history={h:deque(maxlen=250) for h in 'LR'}
        self.valid={h:deque(maxlen=250) for h in 'LR'}
        self.health={h:FrameValidator() for h in 'LR'}
        self.last_invalid={h:-1e12 for h in 'LR'}
        self.port_hands={};self.calibration=None;self.cal_ports=None;self.zero_details=None
        self.take=None;self.last_saved=None;self.status='請正確戴好手套，連接兩個COM。'
        self.monitor=None;self.monitor_count=0;self.monitor_events=None
        self.drops=0;self.take_drops=0;self.errors=[];self.closed=False
        self.log=deque(maxlen=40)
        self.worker=threading.Thread(target=self.work,daemon=True,name='record-and-inference');self.worker.start()

    def call(self,command,value=None,timeout=35):
        response=queue.Queue(maxsize=1)
        self.tasks.put((command,value,response),timeout=1)
        ok,payload=response.get(timeout=timeout)
        if not ok:raise ValueError(payload)
        return payload

    def ready(self,zero=True):
        if len(self.port_hands)!=2 or set(self.port_hands.values())!={'L','R'}:
            raise ValueError('必須收到兩個不同COM的L與R，不能只靠插著兩條線')
        now=self.clock()
        for h in 'LR':
            recent=[p for p in self.history[h] if now-p.received_monotonic<=1]
            if len(recent)<2 or now-recent[-1].received_monotonic>.2:raise ValueError(f'{h}資料不新鮮')
            duration=recent[-1].received_monotonic-recent[0].received_monotonic
            hz=(len(recent)-1)/duration if duration>0 else 0
            if not 80<=hz<=120:raise ValueError(f'{h}為{hz:.1f}Hz，需80–120Hz')
            good=[p for p in self.valid[h] if now-p.received_monotonic<=.6]
            if len(good)<40 or now-self.last_invalid[h]<=.6:raise ValueError(f'{h}資料尚未穩定或有感測異常')
        if zero and (self.calibration is None or self.cal_ports!=self.port_hands):raise ValueError('請先將雙手手掌貼鼓面歸零')

    def zero(self):
        if self.take:raise ValueError('先停止保存，不能在錄製中改歸零')
        self.ready(False);cal={};details={};now=self.clock()
        for h in 'LR':
            packets=[p for p in self.valid[h] if now-p.received_monotonic<=.5]
            if len(packets)<35:raise ValueError('手掌貼鼓面並靜止至少0.5秒')
            a=np.asarray([[p.ax,p.ay,p.az,p.yaw,p.pitch] for p in packets])
            yaw=np.degrees(np.unwrap(np.radians(a[:,3])));mag=np.linalg.norm(a[:,:3],axis=1)
            if mag.std()>.15 or not .65<mag.mean()<1.35 or yaw.std()>3 or a[:,4].std()>3:
                raise ValueError(f'{h}尚未靜止；請手掌貼鼓面，不要握住感測器')
            cal[h]=[float(yaw.mean()%360),float(a[:,4].mean())]
            details[h]=dict(samples=len(packets),magnitude_mean=float(mag.mean()),magnitude_std=float(mag.std()),yaw_std=float(yaw.std()),pitch_std=float(a[:,4].std()))
        self.calibration=cal;self.cal_ports=dict(self.port_hands);self.zero_details=details
        self.status='鼓面歸零完成。拿起鼓棒，再按開始錄製。'
        self.event('zero',json.dumps(dict(calibration=cal,quality=details),ensure_ascii=False))

    def event(self,kind,detail=''):
        if self.monitor_events:
            self.monitor_events.write(json.dumps(dict(t_ms=self.monitor.elapsed_ms,event=kind,detail=detail),ensure_ascii=False)+'\n')
            self.monitor_events.flush()

    def connect(self,ports):
        if self.connections:raise ValueError('已連線；切換COM前請中斷連線')
        if len(ports)!=2 or len(set(ports))!=2 or any(p not in self.ports_provider() for p in ports):raise ValueError('請選兩個不同且存在的COM')
        self.read_stop.clear()
        try:
            for port in ports:self.connections.append(self.serial_factory(port))
        except Exception:
            for c in self.connections:c.close()
            self.connections=[];raise
        self.port_hands={};self.calibration=None;self.cal_ports=None
        for h in 'LR':self.history[h].clear();self.valid[h].clear();self.health[h]=FrameValidator();self.last_invalid[h]=-1e12
        self.monitor=AsyncCsvRecorder(self.data_root/'全程監聽',lambda text:self.log.append(text))
        self.monitor.start('raw_100hz','monitor_live_1023');self.monitor._start_monotonic=self.clock();self.monitor_count=0
        self.monitor_events=self.monitor.path.with_suffix('.events.jsonl').open('w',encoding='utf-8')
        self.event('connect',','.join(ports))
        for conn in self.connections:
            thread=threading.Thread(target=self.read,args=(conn,),daemon=True);thread.start();self.readers.append(thread)
        self.status='已連接，等兩手約100Hz穩定，再手掌貼鼓面歸零。'

    def read(self,conn):
        try:
            while not self.read_stop.is_set():
                line=conn.readline().decode('ascii',errors='replace')
                packet=parse_sensor_line(line)
                if packet:
                    try:self.tasks.put_nowait(('packet',(conn.port,packet),None))
                    except queue.Full:self.drops+=1
        except Exception as exc:
            if not self.read_stop.is_set():
                try:self.tasks.put_nowait(('failure',f'{conn.port}連線中斷：{exc}',None))
                except queue.Full:self.drops+=1

    def save(self):
        if not self.take:return
        take=self.take;self.take=None
        result=take.stop(self.drops-self.take_drops,self.errors)
        result['zero_quality']=self.zero_details
        (take.folder/'session_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        self.last_saved=dict(folder=str(take.folder),report=result)
        self.status=('完整保存' if result['saved_ok'] else '存檔不完整，請勿離場')+f'：Raw {result["raw_written"]}筆'
        if not result['sensor_clean']:self.status+='；感測或封包有異常，請看manifest'
        self.event('take_stop',str(take.folder))

    def disconnect(self):
        self.read_stop.set()
        for c in self.connections:c.close()
        for t in self.readers:t.join(timeout=1)
        self.connections=[];self.readers=[]
        pending=[]
        while True:
            try:item=self.tasks.get_nowait()
            except queue.Empty:break
            if item[0]=='packet':self.packet(*item[1])
            else:pending.append(item)
        for item in pending:self.tasks.put_nowait(item)
        self.save()
        if self.monitor:
            self.event('disconnect');self.monitor.stop()
            report=dict(path=str(self.monitor.path),enqueued=self.monitor.enqueued,written=self.monitor.written,
                dropped=self.monitor.dropped,saved_ok=self.monitor.saved_ok,write_error=self.monitor.write_error,input_queue_dropped=self.drops)
            self.monitor.path.with_suffix('.manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            self.monitor=None;self.monitor_events.close();self.monitor_events=None
        self.port_hands={};self.calibration=None;self.cal_ports=None

    def packet(self,port,p):
        h=p.hand;s=core.packet_sample(p,0,self.calibration[h] if self.calibration else (0,0),0)
        if self.monitor:
            row=core.raw_row(p,s,self.monitor_count);self.monitor_count+=1
            if not self.monitor.enqueue(row,p.received_monotonic):raise RuntimeError('全程監聽寫入失敗')
        if self.take:
            _,reason=self.take.push(p)
            if reason:self.status=f'{h}感測異常：{reason}；Raw仍保存'
        previous=self.port_hands.get(port)
        if previous is not None and previous!=h:raise ValueError('同一COM的HAND_ID改變，停止測試')
        self.port_hands[port]=h;self.history[h].append(p)
        good,reason=self.health[h].check(dict(ax=p.ax,ay=p.ay,az=p.az,yaw=p.yaw,pitch=p.pitch,roll=p.roll))
        if good:self.valid[h].append(p)
        else:self.last_invalid[h]=p.received_monotonic;self.status=f'{h}感測異常：{reason}'

    def snapshot(self,include_ports=True):
        with self.lock:
            now=self.clock();health={}
            for h in 'LR':
                a=[p for p in self.history[h] if now-p.received_monotonic<=1]
                dt=a[-1].received_monotonic-a[0].received_monotonic if len(a)>1 else 0
                health[h]=dict(hz=round((len(a)-1)/dt,1) if dt>0 else 0,
                    fresh=bool(a and now-a[-1].received_monotonic<=.2),invalid=self.health[h].invalid)
            return dict(status=self.status,connected=bool(self.connections),ports=self.ports_provider() if include_ports else [],
                port_hands=dict(self.port_hands),calibration=self.calibration,health=health,
                take=self.take.snapshot() if self.take else None,last_saved=self.last_saved,
                monitor_rows=self.monitor.written if self.monitor else 0,input_queue_dropped=self.drops,
                queued=self.tasks.qsize(),logs=list(self.log),model_name=core.MODEL_NAME,
                model_sha256=core.MODEL_SHA256,selected_dimensions=core.SELECTED_DIMENSIONS,
                detector=self.config,test_mode='direct_frozen_model_test')

    def work(self):
        with threadpool_limits(limits=1):
            while True:
                name,value,response=self.tasks.get()
                try:
                    with self.lock:
                        if name=='quit':
                            self.disconnect();self.closed=True
                            if response:response.put((True,{'closed':True}))
                            return
                        if name=='connect':self.connect(value)
                        elif name=='disconnect':
                            if self.take:raise ValueError('請先停止保存，再中斷COM')
                            self.disconnect();self.status='已中斷連線，監聽檔已保存。'
                        elif name=='packet':self.packet(*value)
                        elif name=='zero':self.zero()
                        elif name=='start':
                            if self.take:raise ValueError('目前已錄製中')
                            self.ready();core.load_components()
                            sid=str(value.get('session','')).strip()
                            if not re.fullmatch(r'[A-Za-z0-9_\-\u4e00-\u9fff]{1,80}',sid):raise ValueError('場次名稱限中英文、數字、底線或連字號')
                            self.errors=[];self.take_drops=self.drops
                            self.take=Take(self.data_root/'模型實測',sid,self.bundle,self.config,self.calibration,self.port_hands,clock=self.clock)
                            self.status='前測已開始，Raw同步保存。照原本每鼓20下測試並同步錄MIDI；不需額外示範。'
                            self.event('take_start',str(self.take.folder))
                        elif name=='stop':self.save()
                        elif name=='marker_start':
                            if not self.take:raise ValueError('先開始新測試，再記錄本項')
                            self.take.begin_item(*value)
                            self.status=f'測試項目已記錄：{value[0]}／{value[1]}；不影響模型判斷'
                        elif name=='marker_end':
                            if not self.take:raise ValueError('目前沒有錄製中的測試')
                            self.take.end_item()
                            self.status='本項結束已記錄；Raw 仍持續保存'
                        elif name=='open_data':os.startfile(str(self.data_root))
                        elif name=='failure':raise RuntimeError(value)
                        elif name=='barrier':pass
                        else:raise ValueError('未知指令')
                        if response:response.put((True,{'ok':True}))
                except Exception as exc:
                    text=f'{type(exc).__name__}: {exc}'
                    self.status=text;self.log.append(text)
                    if self.take and name in ['packet','failure']:
                        self.errors.append(text)
                        try:self.save()
                        except Exception as save_error:self.log.append(str(save_error))
                    if response:response.put((False,text))

    def close(self):
        if not self.closed:self.call('quit',timeout=60);self.worker.join(timeout=3)


def make_server(service,port=0):
    token=secrets.token_urlsafe(24)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,data,content_type='application/json; charset=utf-8'):
            payload=data.encode('utf-8') if isinstance(data,str) else json.dumps(data,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if self.path=='/':
                page=(PACK/'Frontend/第七次_前測主控台.html').read_text(encoding='utf-8').replace('__TOKEN_JSON__',json.dumps(token))
                self.send(200,page,'text/html; charset=utf-8')
            elif self.path=='/api/state':self.send(200,service.snapshot())
            else:self.send(404,{'error':'Not found'})
        def do_POST(self):
            if self.path!='/api/command' or self.headers.get('X-Holo-Token')!=token:
                self.send(403,{'error':'Invalid local token'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('Invalid request length')
                data=json.loads(self.rfile.read(length));command=data['command']
                if command not in {'connect','disconnect','zero','start','stop','open_data','exit'}:
                    raise ValueError('Unknown browser command')
                if command=='exit':
                    service.close();self.send(200,{'closed':True})
                    threading.Thread(target=self.server.shutdown,daemon=True).start()
                else:self.send(200,service.call(command,data.get('value')))
            except Exception as exc:self.send(400,{'error':str(exc)})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    return server


def main():
    # Retain the legacy HTTP harness for historical QA only. Both executable
    # entry points now launch Tk; neither starts HTTP nor opens a browser.
    from seventh_native_gui import main as native_main
    return native_main()


if __name__=='__main__':main()
