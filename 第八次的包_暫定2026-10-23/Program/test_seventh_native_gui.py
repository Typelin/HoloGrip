"""Native Tk integration using injected serial only; all files stay in QA."""
import csv
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import seventh_core as core
import seventh_live as live
import seventh_native_gui as native

OUT=core.PACK/'QA/native_gui_tests'


class Clock:
    def __init__(self):self.now=time.monotonic()
    def __call__(self):return self.now


class FakeSerial:
    opened=[]
    def __init__(self,port):
        assert port in ['QA_LEFT','QA_RIGHT']
        self.port=port;self.done=threading.Event();self.opened.append(port)
    def readline(self):self.done.wait(.02);return b''
    def close(self):self.done.set()


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    formal_before={str(p):core.detector.rt.sha256(p) for p in (core.PACK/'Data').rglob('*') if p.is_file()}
    clock=Clock();base=clock.now;tick=0;checks=[]
    with patch('webbrowser.open',side_effect=AssertionError('Browser must not open')), \
         patch.object(live,'ThreadingHTTPServer',side_effect=AssertionError('HTTP must not start')), \
         patch.object(live.serial,'Serial',side_effect=AssertionError('Real COM must not open')), \
         patch.object(core.runtime,'hit_probability',return_value=1.0):  # synthetic spikes test GUI mechanics, not the hit gate
        service=live.Service(data_root=OUT/'recordings',clock=clock,
            ports_provider=lambda:['QA_LEFT','QA_RIGHT'],serial_factory=FakeSerial)
        app=native.App(service=service);app.withdraw()

        def pump():
            end=time.monotonic()+10
            while app.pending and time.monotonic()<end:
                app.update();time.sleep(.005)
            assert app.pending is None,'Native command timed out'
            app.refresh();app.update_idletasks()
            assert not app.last_error,app.last_error

        def click(key):
            app.refresh();assert not app.buttons[key].instate(['disabled']),key
            app.buttons[key].invoke();pump()

        def feed(n,active='',impulses=()):
            nonlocal tick
            for local in range(1,n+1):
                tick+=1;clock.now=base+tick*.01
                for h,port in [('L','QA_LEFT'),('R','QA_RIGHT')]:
                    p=SimpleNamespace(hand=h,ax=0.,ay=0.,az=6. if h in active and local in impulses else 1.,
                        yaw=25.,pitch=15.,roll=5.,sensor_time_ms=tick*10,packet_id=tick,
                        received_time_ms=100000+tick*10,received_monotonic=clock.now)
                    service.call('packet',(port,p))
            app.refresh();app.update_idletasks()

        try:
            assert [v.get() for v in app.port_vars]==['COM4','COM5']
            app.scan();assert [v.get() for v in app.port_vars]==['COM4','COM5']
            for var,port in zip(app.port_vars,['QA_LEFT','QA_RIGHT']):var.set(port)
            assert app.buttons['start'].instate(['disabled'])
            assert len(app.tiles)==14 and len(app.labels)==2
            assert app.labels['L'].cget('font').endswith('32 bold')
            assert not any('anchor' in key or 'reference' in key for key in app.buttons)
            click('connect');feed(100);click('zero');click('start')
            assert service.snapshot()['take']['phase']=='formal'
            checks.append('COM4/COM5 default retained without fallback; native two-card view, COM/zero gate and immediate formal inference; no browser/HTTP/real COM')

            # This deliberately wrong item label is metadata only.
            app.marker_hand.set('R');app.marker_drum.set('Ride');click('marker_start')
            feed(80,'LR',[30])
            assert app.counts=={'L':1,'R':1}
            app.marker_hand.set('L');app.marker_drum.set('Crash');click('marker_start')
            feed(600,'LR',range(30,571,30))
            assert app.counts=={'L':20,'R':20}
            for h in 'LR':
                assert app.labels[h].cget('text') in core.DRUMS
                assert sum(app.tiles[h,d].cget('bg')=='#a5e8b1' for d in core.DRUMS)==1
                assert '100.0 Hz' in app.hz_labels[h].cget('text')
            checks.append('20 hits per hand, large drum labels, Hz/counters and one active tile per hand')

            low=np.zeros((1,7));low[0,:2]=[.55,.45]
            with patch.object(core.runtime,'probabilities',return_value=low):feed(80,'L',[30])
            assert app.counts=={'L':21,'R':20}
            assert app.labels['L'].cget('text')==str(service.bundle['classes'][0])
            click('stop')
            saved=service.last_saved;report=saved['report'];folder=Path(saved['folder'])
            assert report['saved_ok'] and report['raw_written']==1520,report
            assert report['model_sha256']==core.MODEL_SHA256 and report['selected_dimensions']==104
            assert not report['reference_targets_used_during_inference'] and not report['personal_adaptation']
            assert report['formal_counts']=={'L':21,'R':20}
            assert report['rejected_counts']=={'L':0,'R':0}
            assert len(rows(report['raw_path']))==1520
            predictions=rows(folder/'predictions.csv')
            assert len(predictions)==41 and all(p['accepted']=='True' for p in predictions)
            marks=rows(folder/'markers.csv')
            assert [m['kind'] for m in marks]==['session_start','formal_start','item_start','item_end','item_start','item_end','session_stop']
            assert any(p['hand']=='L' for p in predictions)
            assert app.labels['L'].cget('text')=='本場已保存'
            with patch.object(native.os,'startfile') as opened:
                app.buttons['open'].invoke();opened.assert_called_once_with(str(folder))
            checks.append('1520 Raw rows and 41 predictions saved; close-score output preserved; item switch/stop auto-closes markers; open latest folder')

            # Rendering must not block behind serial close or model validation.
            held=threading.Event();release=threading.Event()
            def hold():
                with service.lock:held.set();release.wait(5)
            holder=threading.Thread(target=hold);holder.start();assert held.wait(2)
            before=time.monotonic();app.refresh();duration=time.monotonic()-before
            release.set();holder.join(2);assert duration<.1,duration
            checks.append('Tk rendering remains nonblocking while the worker lock is held')

            click('start');click('marker_start');feed(80,'L',[30])
            app.close()
            end=time.monotonic()+20
            while not service.closed and time.monotonic()<end:app.update();time.sleep(.01)
            assert service.closed,'Close must stop and save before exit'
            service.worker.join(3)
            assert not service.worker.is_alive()
            assert service.last_saved['report']['saved_ok']
            assert service.last_saved['report']['raw_written']==160
            assert service.last_saved['report']['formal_counts']=={'L':1,'R':0}
            last_marks=rows(Path(service.last_saved['folder'])/'markers.csv')
            assert [m['kind'] for m in last_marks][-2:]==['item_end','session_stop']
            checks.append('Native window close during recording saves final 160 Raw rows, item end and session manifest, closes serial and worker')
        finally:
            service.close()
            try:app.destroy()
            except Exception:pass

    monitors=list((OUT/'recordings/全程監聽').glob('*.manifest.json'))
    monitor=json.loads(max(monitors,key=lambda p:p.stat().st_mtime).read_text(encoding='utf-8'))
    assert monitor['saved_ok'] and monitor['written']==1880,monitor
    assert all(core.detector.rt.sha256(p)==digest for p,digest in formal_before.items())
    result=dict(passed=True,interface='native_tkinter',checks=checks,
        workflow_revision='2026-10-20-venue-free-committee',model_sha256=core.MODEL_SHA256,classifier_contract=core.runtime.CONTRACT,
        first_take_raw_rows=1520,first_take_predictions=41,window_close_raw_rows=160,monitor_rows=1880,
        real_com_opened=False,hardware_tested=False,browser_started=False,http_started=False,
        user_data_preserved=True,qa_files_written_to_formal_data=0)
    (OUT/'native_gui_test.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
