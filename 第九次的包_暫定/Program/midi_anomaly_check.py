from __future__ import annotations
import argparse, csv, json, math, os, sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT=Path(__file__).resolve().parents[1]
PROGRAM=Path(__file__).resolve().parent
sys.path.insert(0,str(PROGRAM))
import midi_label_pipeline as pipeline
from production_core import FrameValidator

DATA=PACKAGE_ROOT/"Data"
MIDI_DIR=DATA/"MIDI"
RAW_DIR=DATA/"Raw"
OUT_DIR=DATA/"對齊與異常報告"

def newest(folder:Path, pats):
    xs=[]
    for pat in pats: xs.extend(folder.glob(pat))
    xs=[p for p in xs if p.is_file()]
    return max(xs,key=lambda p:p.stat().st_mtime) if xs else None

def strict_anomaly_candidates(events, radius_ms=80.0, duplicate_ms=20.0):
    """Match the original 0807 frontend's MIDI-only candidate rules.

    Anomalies are *review candidates*, not proof that an electronic drum
    triggered falsely. Distinct-note dual hits are not anomalous by themselves.
    """
    ordered=sorted(events,key=lambda e:(e.time_ms,e.event_id))
    times=[e.time_ms for e in ordered]
    duplicate_pairs=[]
    duplicate_ids=set()
    crowded_centers=[]
    crowded_ids=set()
    onset_groups=[]
    for i,current in enumerate(ordered):
        if not onset_groups or current.time_ms-onset_groups[-1][0].time_ms>12.0:
            onset_groups.append([current])
        else:
            onset_groups[-1].append(current)
        j=i-1
        while j>=0 and current.time_ms-ordered[j].time_ms<=duplicate_ms:
            previous=ordered[j]
            if current.note==previous.note:
                duplicate_pairs.append({
                    "a_event":previous.event_id,"b_event":current.event_id,
                    "time_ms":round(current.time_ms,3),"gap_ms":round(current.time_ms-previous.time_ms,3),
                    "note":current.note,"zone":current.zone_name,
                })
                duplicate_ids.update((previous.event_id,current.event_id))
            j-=1
    for group in onset_groups:
        center=group[0]
        lo=bisect_left(times,center.time_ms-radius_ms)
        hi=bisect_right(times,center.time_ms+radius_ms)
        if hi-lo>=3:
            crowded_centers.append({"center_event":center.event_id,"time_ms":round(center.time_ms,3),"event_ids":[e.event_id for e in ordered[lo:hi]]})
            crowded_ids.update(e.event_id for e in ordered[lo:hi])
    return {
        "radius_ms":radius_ms,"duplicate_ms":duplicate_ms,
        "same_note_pairs":duplicate_pairs,
        "same_note_affected_event_ids":sorted(duplicate_ids),
        "crowded_centers":crowded_centers,
        "crowded_affected_event_ids":sorted(crowded_ids),
        "anomaly_candidate_event_ids":sorted(duplicate_ids|crowded_ids),
    }

def scan_raw(path:Path):
    by={"L":{"n":0,"bad":Counter(),"ids":[],"times":[]},"R":{"n":0,"bad":Counter(),"ids":[],"times":[]}}
    val={"L":FrameValidator(),"R":FrameValidator()}
    with path.open(encoding="utf-8-sig",newline="") as f:
        rd=csv.DictReader(f)
        fields=set(rd.fieldnames or [])
        for row in rd:
            h=row.get("hand","")
            if h not in by: continue
            def fv(*names):
                for n in names:
                    if n in row and row[n]!="": return float(row[n])
                raise KeyError(names)
            try:
                p={
                    "ax":fv("ax_g","ax"),"ay":fv("ay_g","ay"),"az":fv("az_g","az"),
                    "yaw":fv("yaw_deg","yaw"),"pitch":fv("pitch_deg","pitch"),"roll":fv("roll_deg","roll"),
                    "packet_id":int(float(row.get("packet_id","0"))),
                }
            except Exception:
                continue
            st=by[h];st["n"]+=1;st["ids"].append(p["packet_id"])
            t=row.get("song_time_ms") or row.get("sensor_time_ms")
            if t not in (None,""):
                try:st["times"].append(float(t))
                except:pass
            ok,reason=val[h].check(p)
            if not ok:st["bad"][reason]+=1
    out={}
    for h,st in by.items():
        gaps=sum(max(0,b-a-1) for a,b in zip(st["ids"],st["ids"][1:]) if b>=a)
        duration=(max(st["times"])-min(st["times"]))/1000 if len(st["times"])>=2 else 0
        hz=st["n"]/duration if duration>0 else None
        out[h]={
            "rows":st["n"],"duration_s":round(duration,3),
            "estimated_hz":round(hz,3) if hz else None,
            "missing_packet_ids":int(gaps),
            "invalid_total":int(sum(st["bad"].values())),
            "invalid_reasons":dict(st["bad"]),
        }
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--midi",type=Path)
    ap.add_argument("--raw",type=Path)
    ap.add_argument("--choose",action="store_true",help="現場逐一選檔，避免用最新檔配錯演奏")
    ap.add_argument("--bpm",type=float,default=None,help="進階覆寫；平常不填，使用 MIDI 檔案內建時間資訊")
    ap.add_argument("--output-dir",type=Path,default=OUT_DIR)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    MIDI_DIR.mkdir(parents=True,exist_ok=True);RAW_DIR.mkdir(parents=True,exist_ok=True);OUT_DIR.mkdir(parents=True,exist_ok=True)

    if args.smoke:
        print("FOURTH_MIDI_ANOMALY_SMOKE_PASS");return 0

    midi=args.midi
    raw=args.raw
    if args.choose:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk();root.withdraw()
        if midi is None:
            picked=filedialog.askopenfilename(parent=root,title="選擇這次演奏的原始 MIDI",initialdir=str(MIDI_DIR),filetypes=[("MIDI","*.mid *.midi"),("所有檔案","*.*")])
            midi=Path(picked) if picked else None
        if midi is not None and raw is None:
            picked=filedialog.askopenfilename(parent=root,title="選擇同一次演奏的 Raw CSV；取消＝僅檢查 MIDI",initialdir=str(RAW_DIR),filetypes=[("CSV","*.csv"),("所有檔案","*.*")])
            raw=Path(picked) if picked else None
        root.destroy()
    if midi is None:
        print("[STOP] 未選擇 MIDI，沒有產生報告。");return 2
    if not midi.is_file() or (raw is not None and not raw.is_file()):
        print("[FAIL] 選取的 MIDI 或 Raw 檔案不存在。");return 2
    print("MIDI:",midi)
    print("RAW :",raw if raw else "(尚未找到 Raw)")

    info=pipeline.parse_midi(midi,args.bpm)
    print("TIME:",info["time_mode"],"initial BPM:",round(info["initial_bpm"],3))
    if args.bpm is not None:
        print("[WARN] 本次明確覆寫 BPM；這不是檔案自己的時間軸。")
    ev=list(info["events"])
    strict=strict_anomaly_candidates(ev)
    tick_groups=defaultdict(list)
    for e in ev: tick_groups[e.tick].append(e)
    simultaneous=[g for g in tick_groups.values() if len(g)>1]
    exact_dup=[]
    for tick,g in tick_groups.items():
        c=Counter(e.note for e in g)
        for note,n in c.items():
            if n>1: exact_dup.append({"tick":tick,"note":note,"count":n})

    close=[]
    sev=sorted(ev,key=lambda e:(e.time_ms,e.note))
    for a,b in zip(sev,sev[1:]):
        gap=b.time_ms-a.time_ms
        if gap<100:
            close.append({"a_event":a.event_id,"b_event":b.event_id,"gap_ms":round(gap,3),
                          "a_zone":a.zone_name,"b_zone":b.zone_name,
                          "a_note":a.note,"b_note":b.note})

    low_velocity=[e for e in ev if e.velocity<=5]
    zone_counts=Counter(e.zone_name for e in ev)
    note_counts=Counter(e.note for e in ev)

    raw_report=scan_raw(raw) if raw else None
    problems=[]
    review_notes=[]
    if info["unknown_note_counts"]:problems.append("存在未知 MIDI note")
    if info["timing_warning"]:
        problems.append("MIDI 沒有 tempo 事件，使用格式預設 120 BPM；請向匯出者確認")
    if exact_dup:problems.append("存在同 tick 同 note 重複")
    if strict["same_note_pairs"]:problems.append("同一 MIDI note 20ms 內重複：疑似連續觸發，需對影片/音訊複核")
    if strict["crowded_centers"]:problems.append("onset 群組中心 ±80ms 內至少 3 筆 MIDI：單一訓練窗口過密")
    if simultaneous:review_notes.append("同 tick 多音：可能是真實雙手，不直接算異常")
    if close:review_notes.append("相鄰 <100ms：僅供檢視，不能一律視作異常")
    if low_velocity:review_notes.append("velocity<=5：極弱 MIDI 事件，需複核")
    if raw_report:
        for h,r in raw_report.items():
            if r["invalid_total"]>0:problems.append(f"{h} Raw 有毒 frame {r['invalid_total']}")
            if r["missing_packet_ids"]>0:problems.append(f"{h} Raw packet gap {r['missing_packet_ids']}")
            hz=r["estimated_hz"]
            if hz is not None and not 90<=hz<=110:problems.append(f"{h} Raw Hz 異常 {hz}")

    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    report={
        "generated_at":datetime.now().astimezone().isoformat(),
        "midi":str(midi),"raw":str(raw) if raw else None,"bpm_override":args.bpm,
        "midi_sha256":pipeline._sha256(midi),"time_mode":info["time_mode"],"initial_bpm":info["initial_bpm"],
        "source_note_on_count":info["source_note_on_count"],
        "mapped_event_count":len(ev),
        "unknown_note_counts":info["unknown_note_counts"],
        "excluded_note_counts":info["excluded_note_counts"],
        "mapped_note_counts":dict(note_counts),
        "zone_counts":dict(zone_counts),
        "simultaneous_group_count":len(simultaneous),
        "same_tick_same_note_duplicates":exact_dup,
        "original_frontend_rule":{
            "same_note_duplicate_at_or_under_ms":20,
            "onset_group_tolerance_ms":12,
            "crowded_group_center_radius_ms":80,
            "crowded_min_midi_events":3,
            "meaning":"異常候選＝需人工複核；不同 note 同時出現不直接算錯，MIDI 不含手別",
        },
        "same_note_duplicate_pair_count":len(strict["same_note_pairs"]),
        "same_note_duplicate_affected_event_count":len(strict["same_note_affected_event_ids"]),
        "crowded_window_center_count":len(strict["crowded_centers"]),
        "crowded_window_affected_event_count":len(strict["crowded_affected_event_ids"]),
        "anomaly_candidate_event_count":len(strict["anomaly_candidate_event_ids"]),
        "anomaly_candidate_event_ids":strict["anomaly_candidate_event_ids"],
        "close_onsets_lt100ms_count":len(close),
        "low_velocity_le5_count":len(low_velocity),
        "source_tempo_events":info["tempo_events_in_source"],
        "raw_health":raw_report,
        "problems":problems,
        "review_only_notes":review_notes,
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    out_json=args.output_dir/f"MIDI_Raw_異常報告_{stamp}.json"
    out_csv=args.output_dir/f"MIDI_需複核事件_{stamp}.csv"
    out_json.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    rows=[]
    for d in strict["same_note_pairs"]:
        rows.append({"type":"same_note_within_20ms","time_ms":d["time_ms"],"detail":json.dumps(d,ensure_ascii=False)})
    for d in strict["crowded_centers"]:
        rows.append({"type":"three_or_more_in_plus_minus_80ms","time_ms":d["time_ms"],"detail":json.dumps(d,ensure_ascii=False)})
    for g in simultaneous:
        rows.append({"type":"simultaneous","time_ms":round(g[0].time_ms,3),"detail":" | ".join(f"{e.note}:{e.zone_name}" for e in g)})
    for d in exact_dup:
        rows.append({"type":"same_tick_duplicate","time_ms":"","detail":json.dumps(d,ensure_ascii=False)})
    for d in close:
        rows.append({"type":"close_lt100ms","time_ms":"","detail":json.dumps(d,ensure_ascii=False)})
    for e in low_velocity:
        rows.append({"type":"low_velocity","time_ms":round(e.time_ms,3),"detail":f"note={e.note} {e.zone_name} vel={e.velocity}"})
    with out_csv.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["type","time_ms","detail"]);w.writeheader();w.writerows(rows)

    from midi_csv_alignment_gui import build_midi_only_data, render_midi_only_html
    out_html=args.output_dir/f"MIDI_異常可視化_{midi.stem}_{stamp}.html"
    out_html.write_text(render_midi_only_html(build_midi_only_data(midi,args.bpm),80),encoding="utf-8")

    print("\n=== MIDI ===")
    print("source note-on:",info["source_note_on_count"],"mapped 7-drum:",len(ev))
    print("zone counts:",dict(zone_counts))
    print("unknown notes:",info["unknown_note_counts"])
    print("excluded notes:",info["excluded_note_counts"])
    print("同一 note <=20ms:",len(strict["same_note_pairs"]),"對；受影響事件:",len(strict["same_note_affected_event_ids"]))
    print("onset 群組中心 ±80ms 有至少 3 筆:",len(strict["crowded_centers"]),"個群組；受影響事件:",len(strict["crowded_affected_event_ids"]))
    print("異常候選合計（去重）:",len(strict["anomaly_candidate_event_ids"]),"/",len(ev))
    print("同 tick 多音:",len(simultaneous),"組；<100ms 相鄰:",len(close),"對（僅供檢視，非直接異常）")
    if raw_report:
        print("\n=== RAW ===")
        for h,r in raw_report.items():print(h,r)
    print("\n=== 結論 ===")
    if problems:
        for x in problems:print("[REVIEW]",x)
        print("有 MIDI 候選或未映射 note 待複核；報告已完成，不是程式錯誤。")
    else:
        print("[PASS] 未發現最早前端定義的 MIDI 異常候選；仍需 MIDI 與 Raw / 影片人工確認。")
    for x in review_notes:print("[NOTICE]",x)
    print("報告:",out_json)
    print("事件:",out_csv)
    print("可視化:",out_html)
    if args.choose and hasattr(os,"startfile"):
        os.startfile(str(out_html.resolve()))
    return 1 if problems else 0

if __name__=="__main__":
    raise SystemExit(main())
