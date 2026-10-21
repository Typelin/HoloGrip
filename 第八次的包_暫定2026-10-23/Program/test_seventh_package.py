"""Seventh-package software tests. All generated recordings stay under QA."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mido
import numpy as np
from threadpoolctl import threadpool_limits

import seventh_core as core
import seventh_live as live

PACK=core.PACK
ROOT=PACK.parent
OUT=PACK/'QA/software_tests'
OUT.mkdir(parents=True,exist_ok=True)


class Clock:
    def __init__(self):self.now=time.monotonic()
    def __call__(self):return self.now


def packet(h,base,i,mag=1.,clock=None):
    at=base+i*.01
    if clock:clock.now=at
    return SimpleNamespace(hand=h,ax=0.,ay=0.,az=mag,yaw=25.,pitch=15.,roll=5.,
        sensor_time_ms=i*10,packet_id=i,received_time_ms=100000+i*10,received_monotonic=at)


def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle,cls.config=core.load_components()
        cls.formal_before={str(p):core.detector.rt.sha256(p) for folder in ['Data','今天'] for p in (PACK/folder).rglob('*') if p.is_file()}

    def test_contract_and_independent_stages(self):
        self.assertEqual(set(self.bundle['classes']),set(core.DRUMS))
        self.assertEqual(len(self.bundle['portable']),6)
        for member in self.bundle['portable'].values():
            self.assertEqual(set(member),{'L','R'})
            for h in 'LR':self.assertEqual(member[h]['mlp']['layers'][0]['W'].shape[0],104)
        self.assertFalse(self.bundle['compass_heading_used']);self.assertFalse(self.bundle['pretests_in_training'])
        self.assertLessEqual(self.bundle['future_context_ms'],80)
        self.assertEqual(self.bundle['adaptation']['kind'],'per_drum_prototype_offset')
        self.assertEqual(self.bundle['cymbal_rescue']['strong_g'],2.1)
        self.assertEqual(self.config['refractory_ms'],80.)
        self.assertEqual(self.config['threshold_g'],1.7)
        self.assertEqual(self.bundle['training_events'],13617)
        import inspect
        self.assertEqual(list(inspect.signature(core.Engine.push).parameters),['self','hand','s','raw_frame'])
        self.assertFalse(core.load_guard()['enabled'])
        self.assertEqual(core.detector.rt.sha256(core.MODEL),core.MODEL_SHA256)

    def test_heading_shift_gives_identical_predictions(self):
        rawfile=ROOT/'第五次的包_暫定2026-10-13/Data/Raw/S20261013_P01_song03_take01_raw_100hz_20261013_114237.csv'
        hands=core.trajectory.read_raw(rawfile);rows=readcsv(rawfile);signatures=[]
        with threadpool_limits(limits=1):
            for turn in [0.,90.,-137.]:
                engine=core.Engine(self.bundle,self.config);out=[]
                for _,h,r in sorted((r['row_index'],h,r) for h,a in hands.items() for r in a[:3000]):
                    s=dict(r,absolute_yaw=(r['absolute_yaw']+turn+180)%360-180)
                    frame=dict(ax=s['ax'],ay=s['ay'],az=s['az'],yaw=(float(rows[r['row_index']]['yaw_deg'])+turn+180)%360-180,
                        pitch=float(rows[r['row_index']]['pitch_deg']),roll=s['roll'])
                    pred,reason=engine.push(h,s,frame);self.assertEqual(reason,'');out.extend(pred)
                self.assertTrue(out and all(p['direction_mode']=='venue_free_adaptive' for p in out))
                signatures.append([(p['hand'],p['peak_ms'],p['emitted_ms'],p['drum'],round(p['model_score'],6)) for p in out])
        self.assertTrue(all(sig==signatures[0] for sig in signatures))

    def test_native_live_entry_and_marker_isolation(self):
        import seventh_entry
        with patch.object(seventh_entry.sys,'argv',['seventh_entry.py','live','--smoke']), \
             patch.object(seventh_entry.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(seventh_entry.main(),0)
            self.assertEqual(Path(run.call_args.args[0][1]).name,'seventh_native_gui.py')
        manifest=json.loads((PACK/'package_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['workflow']['interface'],'native_tkinter')
        # The same sensor stream produces identical predictions with or without
        # deliberately different expected-hand/drum marker metadata.
        results=[]
        for marked in [False,True]:
            clock=Clock();take=live.Take(OUT,'QA_MARKER_ISOLATION',self.bundle,self.config,
                {'L':(25,15),'R':(25,15)},{'QA_A':'L','QA_B':'R'},clock=clock);base=clock.now
            if marked:take.begin_item('R','Ride')
            for i in range(100):
                for h in 'LR':take.push(packet(h,base,i,6. if h=='L' and i==40 else 1.,clock))
            take.stop();results.append(readcsv(take.folder/'predictions.csv'))
        stable=lambda r:[{k:v for k,v in p.items() if k!='processing_latency_ms'} for p in r]
        self.assertEqual(stable(results[0]),stable(results[1]))

    def test_recorded_stream_exactly_matches_research(self):
        ref=ROOT/'Research/BestModel_20261021/out/replay_1023'
        recordings=[('pretest_round6',ROOT/'第六次的包_暫定2026-10-16'),('seventh_01',None)]
        import joblib
        cache=joblib.load(ROOT/'第七次的包_暫定2026-10-20/Data/實驗_姿態模型自動標記_20261020/實驗快取.joblib')
        for name,_ in recordings:
            raw=cache['tests'][name.replace('pretest_','pretest:')]['raw']
            hands=core.trajectory.read_raw(raw);source=readcsv(raw);engine=core.Engine(self.bundle,self.config);outputs=[]
            with threadpool_limits(limits=1):
                for _,h,s in sorted((s['row_index'],h,s) for h,a in hands.items() for s in a):
                    row=source[s['row_index']]
                    p,reason=engine.push(h,s,dict(ax=s['ax'],ay=s['ay'],az=s['az'],yaw=float(row['yaw_deg']),pitch=float(row['pitch_deg']),roll=s['roll']))
                    self.assertEqual(reason,'');outputs.extend(p)
            outputs.sort(key=lambda p:(p['peak_ms'],p['hand']))
            reference=readcsv(ref/f'{name}.csv')
            self.assertEqual([(p['hand'],p['peak_ms'],p['emitted_ms'],p['drum']) for p in outputs],
                [(p['hand'],float(p['peak_ms']),float(p['emitted_ms']),p['drum']) for p in reference])
            self.assertTrue(np.allclose([p['model_score'] for p in outputs],[float(p['model_score']) for p in reference]))
            self.assertTrue(all(p['peak_g']>=2.1 or p['drum'] in ('Hi-Hat','Crash','Ride') for p in outputs))

    def test_raw_preserves_invalid_and_multiple_hits(self):
        clock=Clock();take=live.Take(OUT,'QA_RAW_ALL',self.bundle,self.config,{'L':(25,15),'R':(25,15)},
            {'QA_A':'L','QA_B':'R'},clock=clock);base=clock.now
        for i in range(80):
            for h in 'LR':take.push(packet(h,base,i,6. if i in [20,30] else 1.,clock))
        p=packet('L',base,80,1.,clock);p.ax=p.ay=p.az=3.;p.yaw=p.pitch=p.roll=33.75
        got,reason=take.push(p);self.assertEqual(got,[]);self.assertEqual(reason,'i2c_same_word')
        report=take.stop();rows=readcsv(report['raw_path'])
        self.assertTrue(report['saved_ok']);self.assertFalse(report['sensor_clean'])
        self.assertEqual(report['raw_written'],161);self.assertEqual(len(rows),161)
        self.assertEqual(rows[-1]['ax_g'],'3.0');self.assertEqual(report['invalid_by_hand']['L'],1)
        self.assertEqual(report['prediction_count'],4);self.assertEqual(report['formal_counts'],{'L':2,'R':2})
        self.assertTrue(all(p['phase']=='formal' for p in readcsv(take.folder/'predictions.csv')))
        self.assertFalse(report['reference_targets_used_during_inference'])

    def test_direct_start_has_no_demonstration_gate_or_answers(self):
        clock=Clock();take=live.Take(OUT,'QA_DIRECT_TEST',self.bundle,self.config,{'L':(25,15),'R':(25,15)},
            {'QA_A':'L','QA_B':'R'},clock=clock);base=clock.now
        self.assertEqual(take.snapshot()['phase'],'formal')
        self.assertFalse(hasattr(take,'arm_item'));self.assertFalse(hasattr(take,'formal'))
        import inspect
        self.assertNotIn('reference',inspect.signature(live.Take).parameters)
        probabilities=np.zeros((1,7));probabilities[0,list(self.bundle['classes']).index('Ride')]=1
        with patch.object(core.runtime,'probabilities',return_value=probabilities):
            for i in range(80):
                for h in 'LR':take.push(packet(h,base,i,6. if h=='L' and i==30 else 1.,clock))
            self.assertEqual(take.snapshot()['formal_counts'],{'L':1,'R':0})
            self.assertEqual(take.snapshot()['last_hit']['L']['drum'],'Ride')
        report=take.stop();pred=readcsv(take.folder/'predictions.csv');marks=readcsv(take.folder/'markers.csv')
        self.assertEqual(len(pred),1);self.assertTrue(all(p['phase']=='formal' for p in pred))
        self.assertTrue(all(p['drum']=='Ride' for p in pred))
        self.assertEqual(report['formal_counts'],{'L':1,'R':0})
        self.assertEqual(report['raw_written'],160)
        self.assertEqual([m['kind'] for m in marks],['session_start','formal_start','session_stop'])
        self.assertTrue(all(not m['expected_hand'] and not m['expected_drum'] for m in marks))
        self.assertFalse(report['demonstration_required'])
        self.assertEqual(report['test_mode'],'direct_frozen_model_test')
        self.assertFalse(report['personal_adaptation']);self.assertFalse(report['neural_weights_modified'])
        self.assertEqual(report['input_dimensions'],104)
        self.assertEqual(report['selected_dimensions'],104)
        self.assertEqual(report['model_sha256'],core.MODEL_SHA256)
        self.assertFalse(report['output_guard']['enabled'])

    def test_metadata_failure_still_stops_raw(self):
        clock=Clock();take=live.Take(OUT,'QA_CLOSE_ERROR',self.bundle,self.config,{'L':(0,0),'R':(0,0)},
            {'QA_A':'L','QA_B':'R'},clock=clock);base=clock.now
        for i in range(20):
            for h in 'LR':take.push(packet(h,base,i,1.,clock))
        take.pred_file.close()
        report=take.stop()
        self.assertFalse(report['saved_ok']);self.assertEqual(report['raw_written'],40)
        self.assertFalse(take.recorder.active);self.assertTrue(report['errors'])

    def test_close_scores_are_logged_and_emitted_without_guard(self):
        clock=Clock();take=live.Take(OUT,'QA_AMBIGUOUS',self.bundle,self.config,{'L':(25,15),'R':(25,15)},
            {'QA_A':'L','QA_B':'R'},clock=clock);base=clock.now
        probabilities=np.zeros((1,7));probabilities[0,:2]=[.55,.45]
        with patch.object(core.runtime,'probabilities',return_value=probabilities):
            for i in range(80):
                for h in 'LR':take.push(packet(h,base,i,6. if i==30 else 1.,clock))
        snap=take.snapshot();self.assertEqual(snap['formal_counts'],{'L':1,'R':1})
        self.assertEqual(snap['rejected_counts'],{'L':0,'R':0})
        self.assertTrue(snap['last_decision']['L']['accepted']);self.assertIsNotNone(snap['last_hit']['L'])
        report=take.stop();rows=readcsv(take.folder/'predictions.csv')
        self.assertEqual(report['raw_written'],160);self.assertTrue(report['saved_ok'])
        self.assertEqual(report['prediction_count'],2);self.assertEqual(len(rows),2)
        self.assertTrue(all(p['accepted']=='True' and p['guard_reason']=='accepted_no_confidence_filter' for p in rows))
        self.assertTrue(report['rejected_predictions_preserved'])

    def test_sensor_clock_reset_cancels_pending_but_preserves_raw(self):
        clock=Clock();take=live.Take(OUT,'QA_SENSOR_RESET',self.bundle,self.config,{'L':(25,15),'R':(25,15)},
            {'QA_A':'L','QA_B':'R'},clock=clock);base=clock.now
        for i in range(100):
            for h in 'LR':
                p=packet(h,base,i,6. if h=='L' and i in [30,60] else 1.,clock)
                if h=='L' and i==35:p.sensor_time_ms=0
                take.push(p)
        report=take.stop()
        self.assertEqual(report['raw_written'],200)
        self.assertTrue(report['saved_ok']);self.assertFalse(report['sensor_clean'])
        self.assertEqual(report['invalid_by_hand'],{'L':1,'R':0})
        self.assertEqual(report['prediction_count'],1)
        self.assertEqual(float(readcsv(take.folder/'predictions.csv')[0]['peak_ms']),600.)

    def test_collector_raw_only_and_zero_gate(self):
        import seventh_collect as collect
        app=collect.Collection();app.withdraw()
        try:
            app.output_dir=OUT/'collector';app.output_dir.mkdir(exist_ok=True)
            with self.assertRaises(ValueError):app.ready_packets()
            app.serial_connections=[SimpleNamespace(port=p,is_open=True,close=lambda:None) for p in ['QA_A','QA_B']]
            app.observed_ports={'QA_A':'L','QA_B':'R'}
            now=time.monotonic()
            for h in 'LR':
                app.recent[h].extend(packet(h,now-.49,i) for i in range(50))
                app.last_packet_time[h]=time.time()
            with patch.object(collect.messagebox,'showwarning',side_effect=AssertionError('Unexpected gate warning')):
                app._calibrate_both();self.assertEqual(app.zero_ports,app.observed_ports)
                app.session_var.set('QA_NOT_FIELD_COLLECTOR');app._start_recording()
            base=app.recorder._start_monotonic
            for i in range(100):
                for h in 'LR':app._process_packet(packet(h,base,i,7. if i in [20,21,27,60] else 1.))
            app._stop_recording()
            self.assertEqual(app.recorder.written,200);self.assertTrue(app.recorder.saved_ok)
            self.assertEqual(len(readcsv(app.recorder.path)),200)
            meta=json.loads(app.recorder.path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
            self.assertFalse(meta['model_loaded']);self.assertFalse(meta['hit_filter_applied_to_csv'])
        finally:app._on_close()

    def test_bat_real_cmd_smoke(self):
        env={**os.environ,'HOLOGRIP_NO_PAUSE':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'}
        for name,args in [('00_CHECK.bat',[]),('1_測前次.bat',['--smoke']),('2_蒐集這次.bat',['--smoke']),('4_對齊MIDI與CSV.bat',['--smoke'])]:
            content=(PACK/name).read_bytes();self.assertTrue(content.isascii());self.assertIn(b'\r\n',content)
            result=subprocess.run(['cmd.exe','/d','/c',str(PACK/name),*args],env=env,cwd=OUT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=45)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertNotIn('is not recognized',result.stdout+result.stderr)

    def test_midi_only_native_tempo_and_return_codes(self):
        import midi_label_pipeline as pipe
        dest=OUT/'midi';dest.mkdir(exist_ok=True)
        path=dest/'QA_TEMPO.mid';m=mido.MidiFile(ticks_per_beat=480);track=mido.MidiTrack();m.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo',tempo=600000,time=0))
        for note in [40,59]:track.append(mido.Message('note_on',note=note,velocity=90,time=480,channel=9))
        m.save(path);data=pipe.parse_midi(path,None)
        self.assertEqual([e.time_ms for e in data['events']],[600,1200])
        self.assertEqual([e.zone_name for e in data['events']],['小鼓','Ride'])
        env={**os.environ,'HOLOGRIP_NO_PAUSE':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'}
        for expected in [0,1]:
            if expected:track.append(mido.Message('note_on',note=127,velocity=90,time=480));m.save(path)
            result=subprocess.run(['cmd.exe','/d','/c',str(PACK/'3_檢查這次MIDI.bat'),'--midi',str(path),'--output-dir',str(dest)],cwd=OUT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
            self.assertEqual(result.returncode,expected,result.stdout+result.stderr)
            self.assertNotIn('is not recognized',result.stdout+result.stderr)
        self.assertTrue(list(dest.glob('*.html')))

    def test_portable_relative_path_package(self):
        moved=OUT/'搬移測試 空白路徑'/datetime_stamp();moved.mkdir(parents=True)
        for name in ['Program','Model','Frontend','Firmware']:shutil.copytree(PACK/name,moved/name)
        for p in PACK.glob('*.bat'):shutil.copy2(p,moved/p.name)
        shutil.copy2(PACK/'package_manifest.json',moved/'package_manifest.json')
        (moved/'QA').mkdir();(moved/'Data').mkdir()
        result=subprocess.run([sys.executable,str(moved/'Program/seventh_entry.py'),'live','--smoke'],cwd=OUT,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'},capture_output=True,text=True,encoding='utf-8',timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_new_raw_schema_to_alignment_html(self):
        import midi_csv_alignment_gui as gui
        clock=Clock();take=live.Take(OUT,'QA_ALIGNMENT_SCHEMA',self.bundle,self.config,
            {'L':(25,15),'R':(25,15)},{'QA_A':'L','QA_B':'R'},clock=clock)
        base=clock.now
        for i in range(120):
            for h in 'LR':
                hits=[20,60] if h=='L' else [40,80]
                take.push(packet(h,base,i,6. if i in hits else 1.,clock))
        saved=take.stop();raw=Path(saved['raw_path'])
        midi=take.folder/'QA_ALIGNMENT.mid';m=mido.MidiFile(ticks_per_beat=480)
        track=mido.MidiTrack();m.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo',tempo=600000,time=0))
        for note in [40,48,45,43]:track.append(mido.Message('note_on',note=note,velocity=90,time=160,channel=9))
        m.save(midi)
        dest=take.folder/'alignment'
        report=gui.pipeline.build_outputs(raw,midi,dest,None,None,-100,100,80,80)
        self.assertLessEqual(abs(report['alignment']['offset_ms']),10)
        data=gui.build_frontend_data(raw,midi,None,int(report['alignment']['offset_ms']),dest/'midi_events.csv')
        self.assertEqual(len(data['events']),4)
        self.assertEqual([e['mt'] for e in data['events']],[200,400,600,800])
        self.assertEqual(data['events'][0]['name'],'小鼓')
        rendered=gui.render_frontend_html(gui.FRONTEND_TEMPLATE.read_text(encoding='utf-8'),data,False)
        self.assertIn('下載對齊確認紀錄 JSON',rendered)
        self.assertIn('const DATA=',rendered)
        (dest/'QA_MIDI_CSV.html').write_text(rendered,encoding='utf-8')

    def test_original_sources_and_prior_models_unchanged(self):
        path=ROOT/'第六次的包_暫定2026-10-16/Data/四五六聯合神經模型_20261016_v1/來源凍結指紋.json'
        frozen=json.loads(path.read_text(encoding='utf-8'))
        counts={}
        for group,items in frozen.items():
            for name,wanted in items.items():self.assertEqual(core.detector.rt.sha256(Path(name)),wanted,name)
            counts[group]=len(items)
        (PACK/'QA/original_source_integrity.json').write_text(json.dumps(dict(passed=True,counts=counts),indent=2),encoding='utf-8')

    def test_formal_data_preserved(self):
        for path,digest in self.formal_before.items():self.assertEqual(core.detector.rt.sha256(path),digest,path)
        self.assertFalse([p for p in (PACK/'Data').rglob('*') if p.is_file() and 'QA_NOT_FIELD' in p.name])


def datetime_stamp():return str(time.time_ns())


if __name__=='__main__':
    with threadpool_limits(limits=1):
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    report=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
        real_com_test=False,hardware_tested=False,qa_output_root=str(OUT),formal_data_files=len(list((PACK/'Data').rglob('*.csv'))))
    (PACK/'QA/package_tests.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
