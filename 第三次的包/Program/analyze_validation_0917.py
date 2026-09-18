"""Official validation: Raw CSV -> HitDetector -> frozen 7-drum MLP -> MIDI scoring.
Predictions are finalized before MIDI is opened. MIDI never enters inference.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,shutil
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import midi_label_pipeline as midi_pipeline
from product_hit_and_zone import PEAK_LAG_MS,PRODUCT_DETECTOR_KWARGS,WINDOW_HALF_MS,load_zone_model,predict_zone_full,slice_window,window_features
from song_collection_server import HitDetector,SensorPacket

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MODEL=ROOT/'Model'/'hologrip_song2_七鼓點模型_真正驗證版_0826.joblib'
EXPECTED='c2f9ce16750887011219a6f46fee5095c2e4f50c8df3f94548b3c5d806b2f660'
PRED='offline_raw_only_predictions.csv'
MANIFEST='official_raw_only_inference_manifest.json'
REPORT_JSON='official_validation_report.json'
REPORT_MD='official_validation_report.md'

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_csv(path:Path):
    if not path.is_file(): return []
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def ratio(a,b): return a/b if b else 0.0

def marker_time(markers,name):
    for r in markers:
        if r.get('marker')==name:return float(r['session_time_ms'])
    return None

def song_rows(raw,markers):
    rows=[r for r in raw if r.get('phase')=='song']
    if rows:return rows,'phase=song'
    a,b=marker_time(markers,'SONG_START'),marker_time(markers,'SONG_END')
    if a is not None:
        rows=[r for r in raw if float(r.get('session_time_ms',0) or 0)>=a and (b is None or float(r.get('session_time_ms',0) or 0)<=b)]
        if rows:return rows,'SONG_START/SONG_END'
    return raw,'fallback=all raw'

def raw_health(rows):
    out={}
    for hand in ('L','R'):
        p=sorted((r for r in rows if r.get('hand')==hand),key=lambda r:float(r.get('session_time_ms',0) or 0))
        if not p: out[hand]={'rows':0,'status':'MISSING'};continue
        t=[float(r['session_time_ms']) for r in p]; dur=max(1e-6,(t[-1]-t[0])/1000); hz=(len(p)-1)/dur if len(p)>1 else 0
        ids=[int(float(r['packet_id'])) for r in p if r.get('packet_id') not in ('',None)]
        gaps=sum(max(0,b-a-1) for a,b in zip(ids,ids[1:]) if b>=a); resets=sum(b<=a for a,b in zip(ids,ids[1:])); loss=gaps/max(1,len(ids)+gaps)
        zero=sum(int(float(r.get('zero_packet',0) or 0)) for r in p); cal=sum(str(r.get('calibrated','0')).lower() in ('1','true') for r in p)
        out[hand]={'rows':len(p),'mean_hz':hz,'zero_packets':zero,'estimated_packet_gaps':gaps,'estimated_packet_loss_ratio':loss,'packet_id_resets':resets,'calibrated_ratio':cal/len(p),'status':'PASS' if 80<=hz<=120 and zero==0 and loss<=.02 and resets==0 and cal==len(p) else 'WARN'}
    return out

def replay_raw_only(session:Path):
    raw_path=session/'raw_100hz.csv'; markers=read_csv(session/'markers.csv'); raw=read_csv(raw_path)
    if not raw: raise FileNotFoundError(f'找不到或讀不到 {raw_path}')
    if not MODEL.is_file(): raise FileNotFoundError(f'找不到 Frozen model: {MODEL}')
    model_sha=sha(MODEL).lower()
    if model_sha!=EXPECTED: raise RuntimeError(f'Frozen model SHA-256 不符: {model_sha}')
    rows,mode=song_rows(raw,markers)
    rows=sorted(rows,key=lambda r:(float(r.get('session_time_ms',0) or 0),r.get('hand',''),int(float(r.get('packet_id',-1) or -1))))
    clf=load_zone_model(str(MODEL))
    if clf.n_features_in_!=17 or list(clf.classes_)!=list(range(7)): raise RuntimeError('Frozen model 不是 17 維 / 7 類')
    det={h:HitDetector(**PRODUCT_DETECTOR_KWARGS) for h in ('L','R')}; samples={'L':[],'R':[]}; hits=[]
    for r in rows:
        h=r.get('hand','')
        if h not in det:continue
        try:
            t=float(r['session_time_ms']); ax=float(r['ax_g']); ay=float(r['ay_g']); az=float(r['az_g']); yaw=float(r['yaw_deg']); pitch=float(r['pitch_deg']); roll=float(r['roll_deg']); cy=float(r['cal_yaw_deg']); cp=float(r['cal_pitch_deg'])
            pid=int(float(r.get('packet_id',-1) or -1)); stm=int(float(r.get('sensor_time_ms',0) or 0)); host=int(float(r.get('host_time_ms',0) or 0))
        except (KeyError,ValueError,TypeError):continue
        if not all(math.isfinite(v) for v in (t,ax,ay,az,yaw,pitch,roll,cy,cp)):continue
        mag=math.sqrt(ax*ax+ay*ay+az*az); samples[h].append({'song_time_ms':t,'ax':ax,'ay':ay,'az':az,'mag':mag,'yaw':cy,'pitch':cp,'roll':roll})
        pkt=SensorPacket(h,ax,ay,az,yaw,pitch,roll,pid,stm,host,t/1000.0); ev=det[h].add_packet(pkt,cy,cp)
        if ev:hits.append({'hand':h,'trigger':t,'peak':t-PEAK_LAG_MS,'g':float(ev['peak_accel_g'])})
    preds=[]
    for i,d in enumerate(hits,1):
        feat=window_features(slice_window(samples[d['hand']],d['peak'],WINDOW_HALF_MS),d['peak'])
        if feat is None:continue
        drum,p,rank=predict_zone_full(clf,feat)
        preds.append({'prediction_id':i,'hand':d['hand'],'peak_session_ms':round(d['peak'],3),'trigger_session_ms':round(d['trigger'],3),'peak_accel_g':round(d['g'],4),'pred_drum':drum,'pred_proba':round(float(p),6),'second_drum':rank[1][0] if len(rank)>1 else '','second_proba':round(float(rank[1][1]),6) if len(rank)>1 else 0.0,'margin':round(float(rank[0][1]-rank[1][1]),6) if len(rank)>1 else 0.0,'model_sha256':model_sha})
    pred_path=session/PRED; fields=['prediction_id','hand','peak_session_ms','trigger_session_ms','peak_accel_g','pred_drum','pred_proba','second_drum','second_proba','margin','model_sha256']
    with pred_path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(preds)
    manifest={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'official_inference_contract':'raw_100hz.csv -> HitDetector -> peak +/-80ms -> 17 IMU features -> frozen MLP -> 7 drums','midi_used_for_inference':False,'midi_opened_before_prediction_file_finalized':False,'source_raw_csv':str(raw_path),'source_raw_sha256':sha(raw_path),'row_selection':mode,'selected_raw_rows':len(rows),'model_path':str(MODEL),'model_sha256':model_sha,'detector_kwargs':PRODUCT_DETECTOR_KWARGS,'window_half_ms':WINDOW_HALF_MS,'prediction_count':len(preds),'prediction_file':str(pred_path),'prediction_sha256':sha(pred_path)}
    (session/MANIFEST).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return preds,manifest,rows,markers


def slow_test_diagnostic(pred_rows,markers):
    marks=sorted([(float(r['session_time_ms']),r.get('note','').strip()) for r in markers if r.get('marker')=='SLOW_DRUM' and r.get('note')],key=lambda x:x[0])
    song_start=marker_time(markers,'SONG_START'); first=marks[0][0] if marks else None
    if first is None:return {'status':'NOT_RUN','reason':'No SLOW_DRUM markers','detected_predictions':0}
    assigned=[]
    for r in pred_rows:
        t=float(r.get('peak_session_ms',0)); truth=None
        if t<first or (song_start is not None and t>=song_start):continue
        for mt,d in marks:
            if mt<=t:truth=d
            else:break
        if truth:assigned.append((truth,r.get('pred_drum','')))
    per=defaultdict(lambda:{'detected':0,'correct':0,'pred_distribution':Counter()})
    for truth,pred in assigned:per[truth]['detected']+=1;per[truth]['correct']+=int(truth==pred);per[truth]['pred_distribution'][pred]+=1
    out={d:{'detected':x['detected'],'correct':x['correct'],'agreement_on_detected':ratio(x['correct'],x['detected']),'pred_distribution':dict(x['pred_distribution'])} for d,x in per.items()}
    correct=sum(a==b for a,b in assigned)
    return {'status':'OK','detected_predictions':len(assigned),'correct_on_detected':correct,'agreement_on_detected':ratio(correct,len(assigned)),'per_drum':out}

def greedy_temporal_pairs(preds,midi,offset_ms,tol_ms):
    # Compatibility helper for tests/diagnostics using time_ms shaped rows.
    pp=[{'peak_session_ms':x['time_ms'],**x} for x in preds]
    return pairs(pp,midi,offset_ms,tol_ms)

def greedy_exact_count(preds,midi,offset_ms,tol_ms):
    total=0
    for drum in sorted(set([x['drum'] for x in preds]+[x['drum'] for x in midi])):
        total+=len(greedy_temporal_pairs([x for x in preds if x['drum']==drum],[x for x in midi if x['drum']==drum],offset_ms,tol_ms))
    return total
def load_midi(path:Path,bpm:float|None):
    parsed=midi_pipeline.parse_midi(path,float(bpm or 120)); div=int(parsed['ticks_per_beat']); tempos=sorted(parsed.get('tempo_events_in_source',[]),key=lambda x:int(x['tick']))
    if bpm is not None:
        tick_ms=60000/(bpm*div); return [{'id':e.event_id,'time_ms':e.tick*tick_ms,'drum':e.zone_name,'note':e.note} for e in parsed['events']],bpm,'constant_bpm_override',tempos
    def conv(tick):
        last=0; us=500000; total=0.0
        for te in tempos:
            et=int(te['tick'])
            if et>tick:break
            if et>last: total+=(et-last)*us/div;last=et
            us=int(te['microseconds_per_beat'])
        return (total+(tick-last)*us/div)/1000
    events=[{'id':e.event_id,'time_ms':conv(int(e.tick)),'drum':e.zone_name,'note':e.note} for e in parsed['events']]
    nominal=60000000/float(tempos[0]['microseconds_per_beat']) if tempos else 120.0
    return events,nominal,'source_tempo_map',tempos

def pairs(preds,midi,off,tol):
    p=sorted(enumerate(preds),key=lambda x:float(x[1]['peak_session_ms']));m=sorted(enumerate(midi),key=lambda x:float(x[1]['time_ms']));i=j=0;out=[]
    while i<len(p) and j<len(m):
        pi,pr=p[i];mi,ev=m[j];d=float(pr['peak_session_ms'])-(float(ev['time_ms'])+off)
        if d < -tol:i+=1
        elif d > tol:j+=1
        else:out.append((pi,mi,d));i+=1;j+=1
    return out

def auto_offset(preds,midi,center,tol):
    best=(center,-1)
    for off in range(int(center-5000),int(center+5000)+1,10):
        n=len(pairs(preds,midi,float(off),tol))
        if n>best[1]:best=(float(off),n)
    for off in range(int(best[0]-10),int(best[0]+10)+1):
        n=len(pairs(preds,midi,float(off),tol))
        if n>best[1]:best=(float(off),n)
    return best[0]

def score(preds,midi,off,tol):
    ps=pairs(preds,midi,off,tol); correct=0;conf=Counter();per=defaultdict(lambda:{'midi_total':0,'hit_matched':0,'zone_correct':0});errs=[]
    for e in midi:per[e['drum']]['midi_total']+=1
    for pi,mi,d in ps:
        true=midi[mi]['drum'];guess=preds[pi]['pred_drum'];ok=true==guess;correct+=ok;conf[(true,guess)]+=1;errs.append(abs(d));per[true]['hit_matched']+=1;per[true]['zone_correct']+=ok
    hp=ratio(len(ps),len(preds));hr=ratio(len(ps),len(midi));sp=ratio(correct,len(preds));sr=ratio(correct,len(midi));sf=2*sp*sr/(sp+sr) if sp+sr else 0
    return {'midi_events':len(midi),'raw_only_predictions':len(preds),'temporal_matches':len(ps),'hit_precision':hp,'hit_recall':hr,'zone_correct_among_temporal_matches':correct,'zone_accuracy_given_hit':ratio(correct,len(ps)),'strict_correct_events':correct,'strict_precision':sp,'strict_recall':sr,'strict_f1':sf,'mean_abs_time_error_ms':sum(errs)/len(errs) if errs else None,'confusion':[{'true':a,'pred':b,'count':n} for (a,b),n in sorted(conf.items())],'per_drum':dict(per)}
def write_md(r,path):
    e=r['midi_evaluation'];m=r['inference_manifest'];lines=['# HoloGrip 正式 Raw-only Frozen Model 驗證','','**Prediction 先由 Raw CSV 完整產生並寫檔，之後才讀 MIDI。MIDI 不參與 HitDetector、窗口、手別或鼓位判斷。**','',f"- Frozen model SHA-256: `{m['model_sha256']}`",f"- Raw-only predictions: **{m['prediction_count']}**",f"- Raw selection: `{m['row_selection']}`",'','## Raw / COM 健康','']
    for h,x in r['raw_health'].items():lines.append(f"- {h}: **{x.get('status')}** · {x.get('rows',0)} rows · {x.get('mean_hz',0):.1f} Hz · gap {x.get('estimated_packet_gaps',0)} · calibrated {x.get('calibrated_ratio',0)*100:.1f}%")
    if e.get('status')=='OK':
        lines += ['','## 正式結果','',f"- MIDI 真值事件：**{e['midi_events']}**",f"- Raw-only 系統輸出：**{e['raw_only_predictions']}**",f"- HitDetector：Precision **{e['hit_precision']*100:.1f}%** · Recall **{e['hit_recall']*100:.1f}%**",f"- 已抓到事件的七鼓分類：**{e['zone_correct_among_temporal_matches']}/{e['temporal_matches']} = {e['zone_accuracy_given_hit']*100:.1f}%**",f"- **完整事件：Precision {e['strict_precision']*100:.1f}% · Recall {e['strict_recall']*100:.1f}% · F1 {e['strict_f1']*100:.1f}%**",f"- Global MIDI↔Session offset：**{e['offset_ms']:.1f} ms** ({e['alignment_mode']})",f"- 配對容差：±{e['tolerance_ms']:.0f} ms",'','## 每顆鼓','','| 鼓 | MIDI | Hit matched | Zone correct |','|---|---:|---:|---:|']
        for d,x in e['per_drum'].items():lines.append(f"| {d} | {x['midi_total']} | {x['hit_matched']} | {x['zone_correct']} |")
        lines += ['','## 判讀','','- Hit Precision/Recall 低：主要看「有沒有打」偵測。','- Hit 正常但 Zone 低：主要看七鼓模型泛化。','- Strict Precision/Recall/F1 是 Raw → Hit → Drum 的端到端結果。','- 自動對齊只估一個全域 offset；研究報告仍建議用影片或明確同步擊抽查開頭、中段、結尾。']
    else:lines += ['','## MIDI 評分','','尚未提供 MIDI；Raw-only prediction 已先固定完成。']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
def analyze(session:Path,midi_path:Path|None=None,bpm:float|None=None,offset_ms:float|None=None,tol_ms:float=90.0):
    session=session.resolve();preds,manifest,rows,markers=replay_raw_only(session);report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'session':str(session),'inference_contract':'Raw only -> HitDetector -> +/-80ms -> 17 IMU features -> frozen MLP; MIDI only after predictions finalize','inference_manifest':manifest,'raw_health':raw_health(rows)}
    if midi_path is None:report['midi_evaluation']={'status':'NOT_RUN','reason':'No MIDI supplied; raw-only predictions already finalized'}
    else:
        midi_path=midi_path.resolve();midi,used_bpm,mode,tempos=load_midi(midi_path,bpm);center=offset_ms if offset_ms is not None else (marker_time(markers,'SONG_START') or 0.0);off=float(offset_ms) if offset_ms is not None else auto_offset(preds,midi,float(center),tol_ms);metrics=score(preds,midi,off,tol_ms);ref=session/'reference_midi';ref.mkdir(exist_ok=True);copy=ref/midi_path.name
        if copy.resolve()!=midi_path:shutil.copy2(midi_path,copy)
        metrics.update({'status':'OK','midi_path':str(midi_path),'midi_sha256':sha(midi_path),'reference_copy':str(copy),'bpm_used':used_bpm,'tempo_mode':mode,'source_tempo_events':tempos,'alignment_mode':'manual_global_offset' if offset_ms is not None else 'auto_global_offset_search_around_SONG_START','offset_ms':off,'tolerance_ms':tol_ms,'midi_used_for_inference':False});report['midi_evaluation']=metrics
    (session/REPORT_JSON).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');write_md(report,session/REPORT_MD);return report

def main():
    a=argparse.ArgumentParser();a.add_argument('session',type=Path);a.add_argument('--midi',type=Path);a.add_argument('--bpm',type=float);a.add_argument('--offset-ms',type=float);a.add_argument('--tol-ms',type=float,default=90);x=a.parse_args();r=analyze(x.session,x.midi,x.bpm,x.offset_ms,x.tol_ms);e=r['midi_evaluation'];print(x.session/REPORT_MD);print(f"STRICT P {e['strict_precision']*100:.1f}% R {e['strict_recall']*100:.1f}% F1 {e['strict_f1']*100:.1f}%" if e.get('status')=='OK' else 'Raw-only predictions finalized; MIDI not supplied.');return 0
if __name__=='__main__':raise SystemExit(main())

