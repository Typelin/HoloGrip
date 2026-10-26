"""Real browser/HTTP/recording integration with injected serial, never real COM."""
import csv
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from playwright.sync_api import sync_playwright

import seventh_live as live

OUT=live.PACK/'QA/browser_tests'


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


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    clock=Clock();base=clock.now;tick=0
    service=live.Service(data_root=OUT/'recordings',clock=clock,
        ports_provider=lambda:['QA_LEFT','QA_RIGHT'],serial_factory=FakeSerial)
    server=live.make_server(service)
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.05},daemon=True)
    thread.start();url=f'http://127.0.0.1:{server.server_address[1]}/'
    errors=[];checks=[];screens=[]

    def feed(n,active='',impulses=()):
        nonlocal tick
        for local in range(1,n+1):
            tick+=1;clock.now=base+tick*.01
            for h,port in [('L','QA_LEFT'),('R','QA_RIGHT')]:
                mag=6. if h in active and local in impulses else 1.
                p=SimpleNamespace(hand=h,ax=0.,ay=0.,az=mag,yaw=25.,pitch=15.,roll=5.,
                    sensor_time_ms=tick*10,packet_id=tick,received_time_ms=100000+tick*10,
                    received_monotonic=clock.now)
                service.call('packet',(port,p))

    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
            page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(url);page.wait_for_function('state!==null')

            def refresh():page.evaluate('refresh()');page.wait_for_function('!pending')
            def click(selector):
                page.locator(selector).click();page.wait_for_function('!pending');refresh()
                assert not page.locator('#error').inner_text(),page.locator('#error').inner_text()
            def screenshot(name):
                path=OUT/(name+'.png');page.screenshot(path=str(path),full_page=True);screens.append(str(path))
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),name

            assert page.locator('#start').is_disabled()
            assert page.locator('#with-reference, #reference, #reference-arm, #formal').count()==0
            assert '每鼓20下' in page.locator('body').inner_text()
            assert '104' in page.locator('#model-info').inner_text()
            assert live.core.MODEL_SHA256[:12] in page.locator('#model-info').inner_text()
            screenshot('01_initial_desktop')
            checks.append('Initial COM/zero gate; demonstration controls removed')
            click('#connect');feed(100);refresh()
            assert FakeSerial.opened==['QA_LEFT','QA_RIGHT']
            assert service.snapshot()['health']['L']['hz']==100.
            click('#zero');assert page.locator('#start').is_enabled()
            click('#start')
            assert service.snapshot()['take']['phase']=='formal'
            feed(80,'LR',[30]);refresh()
            take=service.snapshot()['take']
            assert take['formal_counts']=={'L':1,'R':1},take
            screenshot('02_first_hit_direct_desktop')
            # Existing 20-hit routine needs no extra phase, targets or item buttons.
            feed(600,'LR',range(30,571,30));refresh()
            take=service.snapshot()['take']
            assert take['formal_counts']=={'L':20,'R':20},take
            for h in 'LR':
                assert page.locator('#result-'+h).inner_text() in live.core.DRUMS
                assert page.locator(f'#kit-{h} .tile.on').count()==1
            screenshot('03_formal_desktop')
            page.set_viewport_size({'width':390,'height':844});refresh()
            screenshot('04_formal_mobile')
            checks.append('First hit immediately classified; 20 hits per hand without a demonstration or count gate; responsive layout')

            low=np.zeros((1,7));low[0,:2]=[.55,.45]
            with patch.object(live.core.runtime,'probabilities',return_value=low):
                feed(80,'L',[30]);refresh()
            assert page.locator('#result-L').inner_text()==str(service.bundle['classes'][0])
            assert page.locator('#kit-L .tile.on').count()==1
            assert '正式輸出 21 下' in page.locator('#count-L').inner_text()
            assert service.snapshot()['take']['formal_counts']=={'L':21,'R':20}
            screenshot('06_close_scores_mobile')
            checks.append('Close scores still produce the frozen-model top class; no legacy margin rejection')

            page.set_viewport_size({'width':1440,'height':1000});click('#stop')
            saved=service.snapshot()['last_saved'];report=saved['report']
            assert report['saved_ok'],report
            assert report['raw_written']==1520,report
            assert not report['demonstration_required']
            assert report['formal_counts']=={'L':21,'R':20}
            assert report['rejected_counts']=={'L':0,'R':0}
            assert report['model_sha256']==live.core.MODEL_SHA256
            assert report['selected_dimensions']==104
            assert not report['personal_adaptation']
            folder=Path(saved['folder'])
            assert all((folder/name).is_file() for name in ['predictions.csv','markers.csv','session_manifest.json'])
            with Path(report['raw_path']).open(encoding='utf-8-sig',newline='') as stream:
                assert len(list(csv.DictReader(stream)))==1520
            with (folder/'predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
                predictions=list(csv.DictReader(stream))
            assert len(predictions)==41 and all(p['phase']=='formal' for p in predictions)
            assert all(p['accepted']=='True' for p in predictions)
            assert page.locator('#saved').inner_text().startswith('完整保存')
            assert page.locator('#result-L').inner_text()=='本場已保存'
            screenshot('05_saved_desktop')
            checks.append('Stop saves all 1520 Raw rows and 41 predictions, recording new model fingerprint and 104 features')

            assert page.request.post(url+'api/command',data={'command':'zero'}).status==403
            for command in ['packet','reference','formal']:
                denied=page.evaluate("""async(command)=>{const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json','X-Holo-Token':TOKEN},body:JSON.stringify({command,value:[]})});return r.status}""",command)
                assert denied==400,command
            checks.append('Local command token; internal and removed demonstration commands rejected')

            click('#start')
            feed(80,'L',[30]);refresh()
            page.once('dialog',lambda dialog:dialog.accept())
            page.locator('#exit').click();page.wait_for_function('closed',timeout=60000)
            thread.join(timeout=5)
            assert not thread.is_alive()
            assert service.closed and service.last_saved['report']['saved_ok']
            assert service.last_saved['report']['raw_written']==160
            assert service.last_saved['report']['formal_counts']=={'L':1,'R':0}
            assert not errors,errors
            checks.append('Exit while active saves final 160 rows and closes serial/monitor/server')
            browser.close()
        monitor_reports=list((OUT/'recordings/全程監聽').glob('*.manifest.json'))
        assert monitor_reports
        monitor=json.loads(max(monitor_reports,key=lambda p:p.stat().st_mtime).read_text(encoding='utf-8'))
        assert monitor['saved_ok'] and monitor['written']==1880,monitor
        formal_files=[p for p in (live.PACK/'Data').rglob('*') if p.is_file()]
        assert not formal_files
        result=dict(passed=True,checks=checks,screenshots=screens,page_errors=errors,
            workflow_revision='2026-10-19-clear-world-219',monitor_rows=monitor['written'],
            real_com_opened=False,hardware_tested=False,formal_data_files=0)
        (OUT/'browser_test.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False))
    finally:
        service.close()
        if thread.is_alive():server.shutdown();thread.join(timeout=5)
        server.server_close()


if __name__=='__main__':main()
