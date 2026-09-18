"""Analyze a HoloGrip Round 3 validation session.

Without MIDI: audits the saved 100 Hz stream and prediction counts.
With MIDI: aligns live hit predictions to the electronic-drum MIDI and reports
hit detection separately from seven-drum classification.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import midi_label_pipeline as midi_pipeline


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists(): return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_ratio(a: int, b: int) -> float:
    return a / b if b else 0.0


def raw_health(rows: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for hand in ("L", "R"):
        part = [r for r in rows if r.get("hand") == hand]
        if not part:
            out[hand] = {"rows": 0, "status": "MISSING"}; continue
        times = [float(r["session_time_ms"]) for r in part]
        duration_s = max(1e-6, (max(times)-min(times))/1000.0)
        hz = (len(part)-1)/duration_s if len(part)>1 else 0.0
        zeros = sum(int(float(r.get("zero_packet",0) or 0)) for r in part)
        ids = [int(float(r["packet_id"])) for r in part if r.get("packet_id") not in (None, "")]
        gaps = sum(max(0,b-a-1) for a,b in zip(ids,ids[1:]) if b>=a)
        resets = sum(1 for a,b in zip(ids,ids[1:]) if b<=a)
        loss = gaps/max(1,len(ids)+gaps)
        calibrated = [r for r in part if str(r.get("calibrated","0")) in ("1","True","true")]
        out[hand] = {
            "rows": len(part), "duration_s": duration_s, "mean_hz": hz,
            "zero_packets": zeros, "zero_ratio": safe_ratio(zeros,len(part)),
            "estimated_packet_gaps": gaps, "estimated_packet_loss_ratio": loss, "packet_id_resets": resets,
            "calibrated_rows": len(calibrated),
            "status": "PASS" if 80 <= hz <= 120 and zeros == 0 and loss <= 0.02 and resets == 0 else "WARN",
        }
    return out


def marker_time(markers: list[dict[str,str]], name: str) -> float | None:
    for row in markers:
        if row.get("marker") == name:
            return float(row["session_time_ms"])
    return None


def slow_test_diagnostic(pred_rows: list[dict[str,str]], markers: list[dict[str,str]]) -> dict[str,Any]:
    drum_marks=sorted(
        [(float(r["session_time_ms"]),r.get("note","").strip()) for r in markers if r.get("marker")=="SLOW_DRUM" and r.get("note")],
        key=lambda x:x[0],
    )
    song_start=marker_time(markers,"SONG_START")
    first_mark=drum_marks[0][0] if drum_marks else None
    slow=[r for r in pred_rows if first_mark is not None and float(r["peak_session_ms"])>=first_mark and (song_start is None or float(r["peak_session_ms"])<song_start)]
    if not drum_marks:
        return {"status":"NOT_RUN","reason":"No SLOW_DRUM markers","detected_predictions":0}
    assigned=[]
    for r in slow:
        t=float(r["peak_session_ms"]); truth=None
        for mt,drum in drum_marks:
            if mt<=t: truth=drum
            else: break
        if truth:
            assigned.append((truth,r["pred_drum"]))
    per=defaultdict(lambda:{"detected":0,"correct":0,"pred_distribution":Counter()})
    for truth,pred in assigned:
        per[truth]["detected"]+=1; per[truth]["correct"]+=int(truth==pred); per[truth]["pred_distribution"][pred]+=1
    serializable={}
    for drum,row in per.items():
        serializable[drum]={"detected":row["detected"],"correct":row["correct"],
                            "agreement_on_detected":safe_ratio(row["correct"],row["detected"]),
                            "pred_distribution":dict(row["pred_distribution"])}
    correct=sum(int(a==b) for a,b in assigned)
    return {"status":"OK","detected_predictions":len(assigned),"correct_on_detected":correct,
            "agreement_on_detected":safe_ratio(correct,len(assigned)),"per_drum":serializable,
            "note":"This does not measure hit recall because the GUI does not know how many physical slow hits were actually played; it measures zone agreement among detected hits."}


def greedy_temporal_pairs(preds: list[dict[str,Any]], midi: list[dict[str,Any]], offset_ms: float, tol_ms: float) -> list[tuple[int,int,float]]:
    p = sorted(enumerate(preds), key=lambda x: x[1]["time_ms"])
    m = sorted(enumerate(midi), key=lambda x: x[1]["time_ms"])
    i=j=0; pairs=[]
    while i<len(p) and j<len(m):
        pi,pr=p[i]; mi,ev=m[j]
        target=ev["time_ms"]+offset_ms
        delta=pr["time_ms"]-target
        if delta < -tol_ms: i+=1
        elif delta > tol_ms: j+=1
        else:
            pairs.append((pi,mi,delta)); i+=1; j+=1
    return pairs


def greedy_exact_count(preds: list[dict[str,Any]], midi: list[dict[str,Any]], offset_ms: float, tol_ms: float) -> int:
    total=0
    drums=sorted(set([x["drum"] for x in preds]+[x["drum"] for x in midi]))
    for drum in drums:
        pp=[x for x in preds if x["drum"]==drum]
        mm=[x for x in midi if x["drum"]==drum]
        total += len(greedy_temporal_pairs(pp,mm,offset_ms,tol_ms))
    return total


def auto_offset(preds: list[dict[str,Any]], midi: list[dict[str,Any]], center: float, tol_ms: float, span_ms: int=5000, step_ms: int=10) -> tuple[float,int]:
    best=(center,-1)
    start=int(round(center-span_ms)); end=int(round(center+span_ms))
    for off in range(start,end+1,step_ms):
        n=len(greedy_temporal_pairs(preds,midi,float(off),tol_ms))
        if n>best[1]: best=(float(off),n)
    fine_start=int(round(best[0]-step_ms)); fine_end=int(round(best[0]+step_ms))
    for off in range(fine_start,fine_end+1):
        n=len(greedy_temporal_pairs(preds,midi,float(off),tol_ms))
        if n>best[1]: best=(float(off),n)
    return best


def load_midi(path: Path, bpm: float | None) -> tuple[list[dict[str,Any]], float | None, list[dict[str,int]], str]:
    # parse_midi gives us reliable note ticks and tempo meta events.  For Round 3
    # we convert ticks with the source tempo map when --bpm is not supplied, so
    # songs with tempo changes do not silently get a wrong time axis.
    parsed=midi_pipeline.parse_midi(path,float(bpm or 120.0))
    tempos=sorted(parsed.get("tempo_events_in_source",[]),key=lambda x:int(x["tick"]))
    division=int(parsed["ticks_per_beat"])
    if bpm is not None:
        tick_ms=60_000.0/(float(bpm)*division)
        events=[{"id":e.event_id,"time_ms":float(e.tick)*tick_ms,"drum":e.zone_name,"note":e.note} for e in parsed["events"]]
        return events,float(bpm),tempos,"constant_bpm_override"

    def tick_to_ms(tick: int) -> float:
        last_tick=0; us_per_beat=500_000; total_us=0.0
        for te in tempos:
            et=int(te["tick"])
            if et>tick: break
            if et>last_tick:
                total_us += (et-last_tick)*us_per_beat/division
                last_tick=et
            us_per_beat=int(te["microseconds_per_beat"])
        total_us += (tick-last_tick)*us_per_beat/division
        return total_us/1000.0

    events=[{"id":e.event_id,"time_ms":tick_to_ms(int(e.tick)),"drum":e.zone_name,"note":e.note} for e in parsed["events"]]
    nominal=(60_000_000.0/float(tempos[0]["microseconds_per_beat"])) if tempos else 120.0
    return events,nominal,tempos,"source_tempo_map"


def analyze(session: Path, midi_path: Path | None, bpm: float | None, offset_ms: float | None, tol_ms: float) -> dict[str,Any]:
    raw=read_csv(session/"raw_100hz.csv")
    pred_rows=read_csv(session/"predictions.csv")
    markers=read_csv(session/"markers.csv")
    preds_all=[{"time_ms":float(r["peak_session_ms"]),"drum":r["pred_drum"],"hand":r.get("hand",""),"phase":r.get("phase","")} for r in pred_rows]
    song_start=marker_time(markers,"SONG_START")
    song_end=marker_time(markers,"SONG_END")
    if song_start is not None:
        song_preds=[p for p in preds_all if p["time_ms"]>=song_start and (song_end is None or p["time_ms"]<=song_end)]
    else:
        song_preds=[p for p in preds_all if p["phase"]=="song"] or preds_all
    report: dict[str,Any]={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "session":str(session), "raw_health":raw_health(raw),
        "prediction_count_all":len(preds_all), "prediction_count_song":len(song_preds),
        "prediction_distribution":dict(Counter(p["drum"] for p in song_preds)),
        "song_start_marker_ms":song_start, "song_end_marker_ms":song_end,
        "slow_test_diagnostic":slow_test_diagnostic(pred_rows,markers),
    }
    if midi_path is None:
        report["midi_evaluation"]={"status":"NOT_RUN","reason":"No MIDI supplied"}
        return report

    midi,used_bpm,tempo_events,tempo_mode=load_midi(midi_path,bpm)
    center=offset_ms if offset_ms is not None else (report["song_start_marker_ms"] or 0.0)
    if offset_ms is None:
        chosen,temporal_n=auto_offset(song_preds,midi,float(center),tol_ms)
        alignment_mode="auto_search_around_song_start"
    else:
        chosen=float(offset_ms); temporal_n=len(greedy_temporal_pairs(song_preds,midi,chosen,tol_ms)); alignment_mode="manual"
    pairs=greedy_temporal_pairs(song_preds,midi,chosen,tol_ms)
    zone_correct=0; confusion=Counter(); abs_errors=[]
    for pi,mi,delta in pairs:
        truth=midi[mi]["drum"]; guess=song_preds[pi]["drum"]
        confusion[(truth,guess)] += 1
        zone_correct += int(truth==guess); abs_errors.append(abs(delta))
    exact=greedy_exact_count(song_preds,midi,chosen,tol_ms)
    hit_recall=safe_ratio(len(pairs),len(midi)); hit_precision=safe_ratio(len(pairs),len(song_preds))
    exact_recall=safe_ratio(exact,len(midi)); exact_precision=safe_ratio(exact,len(song_preds))
    exact_f1=(2*exact_precision*exact_recall/(exact_precision+exact_recall)) if exact_precision+exact_recall else 0.0
    per_true=defaultdict(lambda:{"total":0,"temporal_matched":0,"zone_correct":0})
    for ev in midi: per_true[ev["drum"]]["total"]+=1
    for pi,mi,_ in pairs:
        d=midi[mi]["drum"]; per_true[d]["temporal_matched"]+=1
        if song_preds[pi]["drum"]==d: per_true[d]["zone_correct"]+=1
    report["midi_evaluation"]={
        "status":"OK", "midi_path":str(midi_path), "bpm_used":used_bpm,
        "tempo_mode":tempo_mode, "source_tempo_events":tempo_events, "alignment_mode":alignment_mode,
        "offset_ms":chosen, "tolerance_ms":tol_ms,
        "midi_events":len(midi), "song_predictions":len(song_preds),
        "temporal_matches":len(pairs), "hit_recall":hit_recall, "hit_precision":hit_precision,
        "zone_correct_among_temporal_matches":zone_correct,
        "zone_accuracy_given_hit":safe_ratio(zone_correct,len(pairs)),
        "strict_exact_events":exact, "strict_precision":exact_precision,
        "strict_recall":exact_recall, "strict_f1":exact_f1,
        "mean_abs_time_error_ms":sum(abs_errors)/len(abs_errors) if abs_errors else None,
        "confusion": [{"true":a,"pred":b,"count":n} for (a,b),n in sorted(confusion.items())],
        "per_drum":dict(per_true),
        "warning":"A single global MIDI-to-HoloGrip offset is used. Auto-search estimates only that global offset; keep the original MIDI and verify alignment against SONG_START/video when possible.",
    }
    return report


def write_markdown(report: dict[str,Any], path: Path) -> None:
    lines=["# HoloGrip 第三次驗證分析", "", f"Session: `{report['session']}`", "", "## Raw / COM 健康", ""]
    for h,r in report["raw_health"].items():
        lines.append(f"- {h}: **{r.get('status')}** · rows {r.get('rows',0)} · {r.get('mean_hz',0):.1f} Hz · zero {r.get('zero_packets',0)} · gap {r.get('estimated_packet_gaps',0)}")
    lines += ["", "## Live 模型輸出", "", f"- 全場 predictions: **{report['prediction_count_all']}**", f"- 正式歌曲 predictions: **{report['prediction_count_song']}**", f"- 分布: `{report['prediction_distribution']}`", ""]
    slow=report.get("slow_test_diagnostic",{})
    lines += ["## 七鼓慢打診斷", ""]
    if slow.get("status")=="OK":
        lines.append(f"- 已偵測事件與手動鼓名標記一致：**{slow['correct_on_detected']}/{slow['detected_predictions']} = {slow['agreement_on_detected']*100:.1f}%**")
        lines.append("- 這裡只評估『已偵測事件』的鼓位一致性；因為 GUI 不知道實際敲了幾下，不能拿這一段算 HitDetector Recall。")
        lines += ["", "| 慢打真值 | 已偵測 | 判對 | 一致率 | 預測分布 |", "|---|---:|---:|---:|---|"]
        for drum,r in slow.get("per_drum",{}).items():
            lines.append(f"| {drum} | {r['detected']} | {r['correct']} | {r['agreement_on_detected']*100:.1f}% | `{r['pred_distribution']}` |")
        lines.append("")
    else:
        lines += ["- 尚無 SLOW_DRUM 標記可分析。", ""]
    e=report.get("midi_evaluation",{})
    if e.get("status")!="OK":
        lines += ["## MIDI 對答案", "", "尚未執行。請保留電子鼓原始 MIDI，之後帶入分析腳本。", ""]
    else:
        lines += [
            "## 正式 MIDI 驗證", "",
            f"- BPM used: **{e['bpm_used']:.3f}**",
            f"- 對齊 offset: **{e['offset_ms']:.1f} ms** ({e['alignment_mode']})",
            f"- MIDI 真值事件: **{e['midi_events']}**",
            f"- 系統輸出事件: **{e['song_predictions']}**",
            f"- HitDetector 命中: **{e['temporal_matches']}** · Recall **{e['hit_recall']*100:.1f}%** · Precision **{e['hit_precision']*100:.1f}%**",
            f"- 已命中事件的鼓位分類: **{e['zone_correct_among_temporal_matches']}/{e['temporal_matches']} = {e['zone_accuracy_given_hit']*100:.1f}%**",
            f"- 嚴格時間+鼓位: **{e['strict_exact_events']}** · Precision **{e['strict_precision']*100:.1f}%** · Recall **{e['strict_recall']*100:.1f}%** · F1 **{e['strict_f1']*100:.1f}%**",
            "", "### 每顆鼓", "", "| 鼓 | MIDI | Hit matched | Zone correct |", "|---|---:|---:|---:|",
        ]
        for drum,r in e.get("per_drum",{}).items():
            lines.append(f"| {drum} | {r['total']} | {r['temporal_matched']} | {r['zone_correct']} |")
        lines += ["", "### 判讀規則", "", "- Raw/COM 不健康：先算硬體/座標問題，不把該段直接算成模型失敗。", "- HitDetector 低：處理打擊判斷。", "- HitDetector 正常、zone 低：才是七鼓 MLP 的跨人/跨歌泛化問題。", ""]
    path.write_text("\n".join(lines),encoding="utf-8")


def main() -> int:
    ap=argparse.ArgumentParser(description="Analyze HoloGrip Round 3 validation session")
    ap.add_argument("session",type=Path,help="session folder containing raw_100hz.csv")
    ap.add_argument("--midi",type=Path,help="electronic-drum Standard MIDI file")
    ap.add_argument("--bpm",type=float,help="song BPM; if omitted, use first MIDI tempo meta event when available")
    ap.add_argument("--offset-ms",type=float,help="manual MIDI->session offset; omit to auto-search around SONG_START")
    ap.add_argument("--tol-ms",type=float,default=90.0,help="one-to-one match tolerance, default 90 ms")
    args=ap.parse_args()
    if not args.session.is_dir(): ap.error(f"not a session directory: {args.session}")
    if args.midi is not None and not args.midi.is_file(): ap.error(f"MIDI not found: {args.midi}")
    report=analyze(args.session,args.midi,args.bpm,args.offset_ms,args.tol_ms)
    json_path=args.session/"round3_analysis.json"; md_path=args.session/"round3_analysis.md"
    json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    write_markdown(report,md_path)
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    e=report.get("midi_evaluation",{})
    if e.get("status")=="OK":
        print(f"Hit recall {e['hit_recall']*100:.1f}% | zone given hit {e['zone_accuracy_given_hit']*100:.1f}% | strict F1 {e['strict_f1']*100:.1f}%")
    else:
        print("MIDI not supplied; raw/session health report only.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
