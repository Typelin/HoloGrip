from __future__ import annotations
import argparse, csv, json, math, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT=Path(__file__).resolve().parents[1]
PROGRAM=Path(__file__).resolve().parent
sys.path.insert(0,str(PROGRAM))
import midi_label_pipeline as pipeline
from production_core import FrameValidator

DATA=PACKAGE_ROOT/"Data"
MIDI_DIR=DATA/"第三次MIDI"
RAW_DIR=DATA/"第三次Raw"
OUT_DIR=DATA/"對齊與異常報告"

def newest(folder:Path, pats):
    xs=[]
    for pat in pats: xs.extend(folder.glob(pat))
    xs=[p for p in xs if p.is_file()]
    return max(xs,key=lambda p:p.stat().st_mtime) if xs else None

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
    ap.add_argument("--bpm",type=float,default=110.0)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    MIDI_DIR.mkdir(parents=True,exist_ok=True);RAW_DIR.mkdir(parents=True,exist_ok=True);OUT_DIR.mkdir(parents=True,exist_ok=True)

    if args.smoke:
        print("THIRD_MIDI_ANOMALY_SMOKE_PASS");return 0

    midi=args.midi or newest(MIDI_DIR,("*.mid","*.midi","*.MID","*.MIDI"))
    raw=args.raw or newest(RAW_DIR,("*.csv","*.CSV"))
    if midi is None:
        print(f"[FAIL] {MIDI_DIR} 沒有 MIDI");return 2
    print("MIDI:",midi)
    print("RAW :",raw if raw else "(尚未找到 Raw)")

    info=pipeline.parse_midi(midi,args.bpm)
    ev=list(info["events"])
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
    if info["unknown_note_counts"]:problems.append("存在未知 MIDI note")
    if exact_dup:problems.append("存在同 tick 同 note 重複")
    if len(simultaneous):problems.append("存在同時多音；需確認是真和弦/雙手還是古典/重複")
    if close:problems.append("存在 <100ms 過密 onset；固定單標籤窗口需人工複核")
    if low_velocity:problems.append("存在 velocity<=5 的極弱 MIDI event")
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
        "source_note_on_count":info["source_note_on_count"],
        "mapped_event_count":len(ev),
        "unknown_note_counts":info["unknown_note_counts"],
        "excluded_note_counts":info["excluded_note_counts"],
        "mapped_note_counts":dict(note_counts),
        "zone_counts":dict(zone_counts),
        "simultaneous_group_count":len(simultaneous),
        "same_tick_same_note_duplicates":exact_dup,
        "close_onsets_lt100ms_count":len(close),
        "low_velocity_le5_count":len(low_velocity),
        "source_tempo_events":info["tempo_events_in_source"],
        "raw_health":raw_report,
        "problems":problems,
    }
    out_json=OUT_DIR/f"MIDI_Raw_異常報告_{stamp}.json"
    out_csv=OUT_DIR/f"MIDI_需複核事件_{stamp}.csv"
    out_json.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    rows=[]
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

    print("\n=== MIDI ===")
    print("source note-on:",info["source_note_on_count"],"mapped 7-drum:",len(ev))
    print("zone counts:",dict(zone_counts))
    print("unknown notes:",info["unknown_note_counts"])
    print("excluded notes:",info["excluded_note_counts"])
    print("simultaneous groups:",len(simultaneous),"same-tick duplicates:",len(exact_dup),"<100ms gaps:",len(close))
    if raw_report:
        print("\n=== RAW ===")
        for h,r in raw_report.items():print(h,r)
    print("\n=== 結論 ===")
    if problems:
        for x in problems:print("[REVIEW]",x)
        print("有異常/高密度事件，先複核，不要直接當乾淨訓練標籤。")
    else:
        print("[PASS] 未發現目前規則定義的明顯異常。仍需做 MIDI↔Raw 對齊人工確認。")
    print("報告:",out_json)
    print("事件:",out_csv)
    return 1 if problems else 0

if __name__=="__main__":
    raise SystemExit(main())
