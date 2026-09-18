"""Generate an offline, MIDI-only viewer for the 0812 slow-to-fast test.

The source MIDI remains read-only.  The viewer deliberately distinguishes
near-simultaneous drum chords from sequential onsets, so the latter can be
used to discuss a practical window length for glove training data.
"""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import midi_label_pipeline as pipeline


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
MIDI_PATH = PROJECT_ROOT / "Drum Midi_110BPM (0812)_慢到快.mid"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "Derived" / "Song_Collection_COM" / "S20260812_P01_慢到快_MIDI測速"
OUTPUT_HTML = OUTPUT_DIR / "HoloGrip_慢到快MIDI節奏檢視_0817_ZH_TW.html"
OUTPUT_JSON = OUTPUT_DIR / "慢到快MIDI間隔分析_0817_ZH_TW.json"
OUTPUT_REPORT = PROJECT_ROOT / "outputs" / "慢到快MIDI間隔分析_0817_ZH_TW.md"

BPM = 110.0
SIMULTANEOUS_TOLERANCE_MS = 20.0


def fmt_ms(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.3f} ms"


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def group_onsets(events: list[pipeline.MidiNoteEvent]) -> list[dict[str, Any]]:
    """Group MIDI note-ons that share the same physical onset (within 20 ms)."""
    groups: list[dict[str, Any]] = []
    for event in events:
        group = groups[-1] if groups else None
        if group is None or event.time_ms - group["time_ms"] > SIMULTANEOUS_TOLERANCE_MS:
            group = {"group_id": len(groups) + 1, "time_ms": event.time_ms, "events": []}
            groups.append(group)
        group["events"].append(event)

    for index, group in enumerate(groups):
        previous = groups[index - 1]["time_ms"] if index else None
        following = groups[index + 1]["time_ms"] if index + 1 < len(groups) else None
        group["previous_gap_ms"] = group["time_ms"] - previous if previous is not None else None
        group["next_gap_ms"] = following - group["time_ms"] if following is not None else None
    return groups


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def build_data() -> tuple[dict[str, Any], dict[str, Any]]:
    # Preserve the source event that uses note 40. It is not in the confirmed
    # seven-zone hand map, but hiding it would create a false gap in this
    # MIDI-only timing inspection. It therefore receives its own visible lane.
    standard_midi = pipeline.parse_midi(MIDI_PATH, BPM)
    original_zone_map = dict(pipeline.ZONE_BY_NOTE)
    try:
        pipeline.ZONE_BY_NOTE[40] = (7, "未映射 note 40")
        midi = pipeline.parse_midi(MIDI_PATH, BPM)
    finally:
        pipeline.ZONE_BY_NOTE.clear()
        pipeline.ZONE_BY_NOTE.update(original_zone_map)
    events = midi["events"]
    if not events:
        raise ValueError("MIDI does not contain mapped hand drum events.")

    groups = group_onsets(events)
    event_rows = [
        {
            "event_id": event.event_id,
            "time_ms": round(event.time_ms, 3),
            "note": event.note,
            "zone_id": event.zone_id,
            "zone_name": event.zone_name,
            "velocity": event.velocity,
            "group_id": next(group["group_id"] for group in groups if event in group["events"]),
        }
        for event in events
    ]
    group_rows = [
        {
            "group_id": group["group_id"],
            "time_ms": round(group["time_ms"], 3),
            "previous_gap_ms": round(group["previous_gap_ms"], 3) if group["previous_gap_ms"] is not None else None,
            "next_gap_ms": round(group["next_gap_ms"], 3) if group["next_gap_ms"] is not None else None,
            "events": [
                {
                    "event_id": event.event_id,
                    "time_ms": round(event.time_ms, 3),
                    "note": event.note,
                    "zone_name": event.zone_name,
                    "velocity": event.velocity,
                }
                for event in group["events"]
            ],
        }
        for group in groups
    ]
    raw_pairs = [
        {
            "from_event_id": previous.event_id,
            "to_event_id": current.event_id,
            "from_note": previous.note,
            "to_note": current.note,
            "from_zone": previous.zone_name,
            "to_zone": current.zone_name,
            "time_ms": round(current.time_ms, 3),
            "gap_ms": round(current.time_ms - previous.time_ms, 3),
        }
        for previous, current in zip(events, events[1:])
    ]
    same_note_pairs = [
        pair for pair in raw_pairs if pair["from_note"] == pair["to_note"]
    ]
    group_intervals = [
        {
            "to_group_id": group["group_id"],
            "time_ms": round(group["time_ms"], 3),
            "gap_ms": round(group["previous_gap_ms"], 3),
            "event_count": len(group["events"]),
        }
        for group in groups[1:]
    ]
    min_raw = min(raw_pairs, key=lambda pair: pair["gap_ms"]) if raw_pairs else None
    min_same_note = min(same_note_pairs, key=lambda pair: pair["gap_ms"]) if same_note_pairs else None
    min_group = min(group_intervals, key=lambda item: item["gap_ms"]) if group_intervals else None
    group_gaps = [item["gap_ms"] for item in group_intervals]
    note_counts = Counter(event.zone_name for event in events)
    event_by_group = {row["event_id"]: row["group_id"] for row in event_rows}
    simultaneous_groups = [group for group in group_rows if len(group["events"]) > 1]

    summary = {
        "source_file": str(MIDI_PATH),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bpm_override": BPM,
        "ticks_per_beat": midi["ticks_per_beat"],
        "source_tempo_event_count": len(standard_midi["tempo_events_in_source"]),
        "duration_ms": round(midi["track_end_ms"], 3),
        "source_note_on_count": midi["source_note_on_count"],
        "displayed_note_event_count": len(events),
        "mapped_hand_drum_event_count": sum(event.zone_id < 7 for event in events),
        "unmapped_display_event_count": sum(event.zone_id >= 7 for event in events),
        "excluded_note_counts": standard_midi["excluded_note_counts"],
        "unknown_note_counts": standard_midi["unknown_note_counts"],
        "zone_counts": dict(sorted(note_counts.items())),
        "simultaneous_tolerance_ms": SIMULTANEOUS_TOLERANCE_MS,
        "onset_group_count": len(groups),
        "simultaneous_group_count": len(simultaneous_groups),
        "simultaneous_event_count": sum(len(group["events"]) for group in simultaneous_groups),
        "minimum_raw_adjacent_gap": min_raw,
        "minimum_same_note_adjacent_gap": min_same_note,
        "minimum_sequential_onset_gap": min_group,
        "median_sequential_onset_gap_ms": round(median(group_gaps), 3) if group_gaps else None,
        "p10_sequential_onset_gap_ms": round(percentile(group_gaps, 0.10), 3) if group_gaps else None,
        "p90_sequential_onset_gap_ms": round(percentile(group_gaps, 0.90), 3) if group_gaps else None,
    }
    viewer_data = {
        "summary": summary,
        "events": event_rows,
        "groups": group_rows,
        "intervals": group_intervals,
        "event_group_map": event_by_group,
        "note_colors": {
            "小鼓": "#e0a106",
            "高音 Tom": "#2f9e8f",
            "中音 Tom": "#5b8cce",
            "落地 Tom": "#d96d4c",
            "Hi-Hat": "#c5b651",
            "Crash": "#e48b47",
            "Ride": "#78a1a8",
            "未映射 note 40": "#8c8f90",
        },
    }
    return summary, viewer_data


def report_markdown(summary: dict[str, Any]) -> str:
    raw = summary["minimum_raw_adjacent_gap"]
    same = summary["minimum_same_note_adjacent_gap"]
    sequential = summary["minimum_sequential_onset_gap"]
    return f"""# 慢到快 MIDI 間隔分析｜0817

## 資料來源

- 檔案：`{summary['source_file']}`
- 解析基準：檔名與現場指定為固定 110 BPM；來源 MIDI 沒有 tempo event，因此毫秒由 110 BPM 與 tick 換算。若流音確認 BPM 不同，所有毫秒值需等比例重算。僅統計 7 種手部鼓點，腳部大鼓與 Hi-Hat 腳踏會列為排除項。
- MIDI note-on：原始 {summary['source_note_on_count']} 筆；其中確認為 7 種手部鼓點 {summary['mapped_hand_drum_event_count']} 筆；另有 {summary['unmapped_display_event_count']} 筆 note 40 不在目前映射表，已另列時間軸而未刪除；onset 群組 {summary['onset_group_count']} 個。

## 最小間隔

- 原始相鄰 MIDI 最小間隔：{fmt_ms(raw['gap_ms']) if raw else '--'}，事件 #{raw['from_event_id']} {raw['from_zone']} → #{raw['to_event_id']} {raw['to_zone']}。本檔沒有同時 onset 群組，因此此值是連續兩次 MIDI 觸發，不是同時雙鼓。
- 同一鼓點連續兩擊最小間隔：{fmt_ms(same['gap_ms']) if same else '--'}，事件 #{same['from_event_id']} → #{same['to_event_id']}。本檔該最小值是小鼓連打；它是觀測到的 MIDI 觸發下限，仍不應宣稱為人體生理絕對上限。
- 合併 20 ms 內同時 onset 後，最小連續敲擊間隔：{fmt_ms(sequential['gap_ms']) if sequential else '--'}，到達群組 #{sequential['to_group_id'] if sequential else '--'}。**這才是本次慢到快測試可用來討論手部連續打擊上限的主要數字。**
- 群組間隔中位數：{fmt_ms(summary['median_sequential_onset_gap_ms'])}；最快 10% 的門檻（P10）：{fmt_ms(summary['p10_sequential_onset_gap_ms'])}。

## 判讀規則

- 20 ms 內出現多筆不同鼓點，列為同一個「同時 onset 群組」，代表可能的雙手同時打擊，不視為兩次連續手部動作。
- 兩個群組之間的 gap 才是連續兩次打擊可用的時間距離；採樣窗口半徑要小於該距離的一半，才不會大量吃到下一次打擊。
- 這份 MIDI 只能回答事件時間與鼓點種類，不能自行判定左手或右手。
"""


def render_html(viewer_data: dict[str, Any]) -> str:
    data_json = safe_json(viewer_data)
    source_name = html.escape(Path(viewer_data["summary"]["source_file"]).name)
    source_path = html.escape(viewer_data["summary"]["source_file"])
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HoloGrip 慢到快 MIDI 節奏檢視｜0817</title>
<style>
  :root {{ color-scheme: dark; --canvas:#0b0c0e; --panel:#14161a; --panel2:#1b1e23; --paper:#f2efe8; --ink:#1a1b1e; --muted:#a8aaa9; --line:rgba(242,239,232,.11); --amber:#e0a106; --teal:#2f9e8f; --danger:#d96d4c; --blue:#5b8cce; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--canvas); color:var(--paper); font-family:"IBM Plex Sans","Noto Sans TC",system-ui,sans-serif; letter-spacing:0; }}
  main {{ width:min(1180px,calc(100% - 32px)); margin:0 auto; padding:28px 0 54px; }}
  header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:20px; border-bottom:1px solid var(--line); padding-bottom:20px; }}
  h1,h2,p {{ margin:0; }} h1 {{ font-size:clamp(25px,4vw,38px); font-weight:700; line-height:1.12; }} h2 {{ font-size:18px; font-weight:600; }}
  .eyebrow,.meta,.label {{ color:var(--muted); font:12px/1.45 "IBM Plex Mono","Cascadia Mono",monospace; letter-spacing:.04em; }}
  .eyebrow {{ color:var(--amber); text-transform:uppercase; margin-bottom:8px; }} .meta {{ max-width:500px; overflow-wrap:anywhere; text-align:right; }}
  .notice {{ border-left:3px solid var(--amber); background:rgba(224,161,6,.08); padding:14px 16px; margin:24px 0; line-height:1.6; color:#dedbd2; }}
  .stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); border-radius:8px; overflow:hidden; background:var(--panel); }}
  .stat {{ padding:16px; min-height:108px; border-right:1px solid var(--line); }} .stat:last-child {{ border-right:0; }}
  .stat strong {{ display:block; font:700 clamp(24px,3vw,34px)/1.1 "IBM Plex Sans",sans-serif; color:var(--amber); margin:7px 0 6px; font-variant-numeric:tabular-nums; }}
  .stat small {{ display:block; color:var(--muted); line-height:1.45; }}
  .section {{ margin-top:32px; }} .section-head {{ display:flex; gap:16px; justify-content:space-between; align-items:flex-end; margin-bottom:12px; }}
  .section-head p {{ color:var(--muted); font-size:13px; line-height:1.45; max-width:710px; }}
  .plot-wrap {{ border:1px solid var(--line); border-radius:8px; background:var(--panel); overflow:hidden; }}
  svg {{ display:block; width:100%; height:auto; background:linear-gradient(90deg,rgba(255,255,255,.012),transparent 20%,transparent 80%,rgba(255,255,255,.012)); }}
  .controls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:14px 0 12px; }}
  button,select {{ min-height:36px; border:1px solid var(--line); border-radius:6px; background:var(--panel2); color:var(--paper); padding:7px 11px; font:13px "IBM Plex Sans","Noto Sans TC",sans-serif; cursor:pointer; }}
  button:hover,button:focus-visible,select:focus-visible {{ border-color:var(--amber); outline:0; }} button.primary {{ background:var(--amber); color:var(--ink); border-color:var(--amber); font-weight:700; }}
  .selected {{ margin-left:auto; color:var(--muted); font:12px/1.4 "IBM Plex Mono","Cascadia Mono",monospace; }}
  .detail {{ display:grid; grid-template-columns:1.2fr .8fr; gap:12px; }} .detail-box {{ border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:16px; min-height:126px; }}
  .detail-title {{ display:flex; justify-content:space-between; gap:12px; margin-bottom:12px; align-items:baseline; }} .detail-title strong {{ color:var(--amber); font:600 16px "IBM Plex Sans",sans-serif; }}
  .events {{ display:flex; flex-wrap:wrap; gap:7px; }} .event-chip {{ border-radius:4px; color:var(--ink); padding:5px 7px; font-size:12px; font-weight:700; }}
  .event-chip small {{ font-weight:400; }} .detail-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }} .detail-value {{ display:block; font:600 19px/1.2 "IBM Plex Mono","Cascadia Mono",monospace; color:var(--paper); margin-top:4px; }}
  .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:8px; }} table {{ width:100%; border-collapse:collapse; font-size:13px; }} th,td {{ text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }} th {{ background:var(--panel2); color:var(--muted); font:11px "IBM Plex Mono","Cascadia Mono",monospace; letter-spacing:.04em; }} tbody tr {{ cursor:pointer; }} tbody tr:hover,tbody tr.active {{ background:rgba(224,161,6,.1); }} tbody tr:last-child td {{ border-bottom:0; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:10px 16px; padding:10px 12px 0; color:var(--muted); font-size:12px; }} .legend span::before {{ content:""; display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--swatch); margin-right:5px; }}
  .footer {{ margin-top:20px; color:var(--muted); font-size:12px; line-height:1.55; }}
  @media (max-width:760px) {{ main {{ width:min(100% - 20px,1180px); padding-top:18px; }} header,.section-head {{ align-items:flex-start; flex-direction:column; }} .meta {{ text-align:left; }} .stats {{ grid-template-columns:1fr 1fr; }} .stat:nth-child(2) {{ border-right:0; }} .stat:nth-child(-n+2) {{ border-bottom:1px solid var(--line); }} .detail {{ grid-template-columns:1fr; }} .selected {{ width:100%; margin-left:0; }} }}
</style>
</head>
<body>
<main>
  <header>
    <div><p class="eyebrow">HoloGrip · MIDI only · offline</p><h1>慢到快 MIDI 節奏檢視</h1></div>
    <p class="meta">預載檔案：{source_name}<br>{source_path}</p>
  </header>
  <p class="notice">這頁只讀取已內嵌的 MIDI note-on 時間與鼓點種類，沒有 CSV、影片或音訊。<strong>20 ms 內的多個鼓點會視為同一個同時 onset 群組</strong>；真正用來看連續敲擊速度的是群組與群組之間的間隔。此檔未帶 tempo event，毫秒依檔名／現場指定的 110 BPM 換算。若來源出現非七鼓點映射的 note，仍會用獨立軌顯示，避免時間軸漏點。</p>
  <section class="stats" aria-label="測速摘要">
    <div class="stat"><span class="label">原始 MIDI note-on</span><strong id="metric-events">--</strong><small>含未映射 note；不把腳部事件算入。</small></div>
    <div class="stat"><span class="label">同時 onset 群組</span><strong id="metric-groups">--</strong><small>20 ms 內合併為同一次打擊時機。</small></div>
    <div class="stat"><span class="label">最小原始相鄰間隔</span><strong id="metric-raw">--</strong><small>本檔無同時 onset，故等同連續觸發。</small></div>
    <div class="stat"><span class="label">最小連續敲擊間隔</span><strong id="metric-sequential">--</strong><small>合併同時 onset 後的主要測速值。</small></div>
  </section>

  <section class="section">
    <div class="section-head"><div><p class="eyebrow">01 · interval profile</p><h2>由慢到快的連續敲擊間隔</h2></div><p>每一點代表一個 onset 群組與上一個群組的距離。點越低，表示越接近下一次打擊。選取點可同步檢視下方局部時間軸。</p></div>
    <div class="plot-wrap"><svg id="interval-chart" viewBox="0 0 1100 330" role="img" aria-label="每個 MIDI onset 群組的相鄰間隔圖"></svg><div class="legend" id="interval-legend"></div></div>
  </section>

  <section class="section">
    <div class="section-head"><div><p class="eyebrow">02 · selected onset</p><h2>選取 onset 的 MIDI 局部時間軸</h2></div><p>這是 MIDI 原始時間軸，不是加速度曲線。每條橫列是一種鼓點；中央虛線是選取的 onset 群組。</p></div>
    <div class="controls" aria-label="MIDI onset 導覽">
      <button type="button" data-action="first">第一個</button><button type="button" data-action="previous">上一個</button><button class="primary" type="button" data-action="fastest">跳到最快</button><button type="button" data-action="next">下一個</button><button type="button" data-action="last">最後一個</button>
      <label class="label" for="span-select">檢視範圍</label><select id="span-select" aria-label="局部時間軸範圍"><option value="1500">±1.5 秒</option><option value="3000" selected>±3 秒</option><option value="6000">±6 秒</option></select>
      <span class="selected" id="selected-summary">--</span>
    </div>
    <div class="plot-wrap"><svg id="timeline-chart" viewBox="0 0 1100 370" role="img" aria-label="選取 MIDI onset 周圍的各鼓點事件時間軸"></svg></div>
  </section>

  <section class="section detail">
    <div class="detail-box"><div class="detail-title"><span class="label">選取群組內容</span><strong id="detail-group">--</strong></div><div class="events" id="detail-events"></div></div>
    <div class="detail-box"><div class="detail-grid"><div><span class="label">距離前一個 onset</span><span class="detail-value" id="detail-previous">--</span></div><div><span class="label">距離下一個 onset</span><span class="detail-value" id="detail-next">--</span></div></div><p class="footer" id="detail-note"></p></div>
  </section>

  <section class="section">
    <div class="section-head"><div><p class="eyebrow">03 · event list</p><h2>最快的連續 onset 間隔</h2></div><p>清單按群組間隔由短到長排序；點選任何一列會跳至該 onset。</p></div>
    <div class="table-wrap"><table><thead><tr><th>群組</th><th>到達時間</th><th>距上一群組</th><th>本群組鼓點</th></tr></thead><tbody id="fastest-table"></tbody></table></div>
  </section>
  <p class="footer">判讀提醒：MIDI 能確認何時觸發何種鼓點，不能從檔案本身辨識左右手。若要設定手套資料切片窗口，應以「最小連續敲擊間隔」作上限參考，並另外保留前後緩衝，而不是直接使用原始事件最小值。</p>
</main>
<script id="midi-data" type="application/json">{data_json}</script>
<script>
(() => {{
  'use strict';
  const D = JSON.parse(document.getElementById('midi-data').textContent);
  const S = D.summary;
  const NS = 'http://www.w3.org/2000/svg';
  let selectedIndex = Math.max(0, D.groups.length - 1);
  let localSpanMs = 3000;
  const fmt = value => value == null || !Number.isFinite(value) ? '--' : `${{value.toFixed(3)}} ms`;
  const time = value => {{ const sec = value / 1000; const min = Math.floor(sec / 60); return `${{min}}:${{(sec % 60).toFixed(3).padStart(6,'0')}}`; }};
  const notes = events => events.map(event => `${{event.zone_name}} (note ${{event.note}})`).join(' · ');
  const element = (tag, attrs = {{}}, text = '') => {{ const node = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([key,value]) => node.setAttribute(key, String(value))); if(text) node.textContent = text; return node; }};
  const clear = node => {{ while(node.firstChild) node.removeChild(node.firstChild); }};
  const scale = (value, domainStart, domainEnd, rangeStart, rangeEnd) => rangeStart + (value-domainStart) / (domainEnd-domainStart || 1) * (rangeEnd-rangeStart);
  const color = zone => D.note_colors[zone] || '#d4d0c6';

  document.getElementById('metric-events').textContent = S.displayed_note_event_count;
  document.getElementById('metric-groups').textContent = S.onset_group_count;
  document.getElementById('metric-raw').textContent = fmt(S.minimum_raw_adjacent_gap?.gap_ms).replace(' ms','');
  document.getElementById('metric-sequential').textContent = fmt(S.minimum_sequential_onset_gap?.gap_ms).replace(' ms','');
  document.getElementById('interval-legend').innerHTML = `<span style="--swatch:#e0a106">實線：群組間隔</span><span style="--swatch:#2f9e8f">綠點：目前選取</span><span style="--swatch:#d96d4c">紅點：最快群組間隔</span>`;

  function drawIntervalChart() {{
    const svg = document.getElementById('interval-chart'); clear(svg);
    const W=1100,H=330,L=66,R=26,T=24,B=55;
    const values = D.intervals.map(item => item.gap_ms);
    const upper = Math.max(120, Math.ceil((Math.max(...values) || 120) / 100) * 100);
    const x = index => scale(index, 0, Math.max(D.intervals.length-1,1), L, W-R);
    const y = value => scale(value, 0, upper, H-B, T);
    [0,.25,.5,.75,1].forEach(fraction => {{ const value=upper*fraction; const yy=y(value); svg.append(element('line',{{x1:L,y1:yy,x2:W-R,y2:yy,stroke:'rgba(242,239,232,.12)','stroke-width':1}})); svg.append(element('text',{{x:L-10,y:yy+4,fill:'#a8aaa9','font-size':12,'text-anchor':'end'}},`${{Math.round(value)}} ms`)); }});
    const path = D.intervals.map((item,index)=>`${{index?'L':'M'}} ${{x(index).toFixed(2)}} ${{y(item.gap_ms).toFixed(2)}}`).join(' ');
    svg.append(element('path',{{d:path,fill:'none',stroke:'#e0a106','stroke-width':2.2}}));
    const fastestGroup = S.minimum_sequential_onset_gap?.to_group_id;
    D.intervals.forEach((item,index) => {{ const active=item.to_group_id===D.groups[selectedIndex].group_id; const fastest=item.to_group_id===fastestGroup; const circle=element('circle',{{cx:x(index),cy:y(item.gap_ms),r:active?6:fastest?4.5:3.2,fill:active?'#2f9e8f':fastest?'#d96d4c':'#e0a106',stroke:'#0b0c0e','stroke-width':active?2:1,'data-group':item.to_group_id}}); circle.style.cursor='pointer'; circle.addEventListener('click',()=>selectGroup(item.to_group_id-1)); svg.append(circle); }});
    svg.append(element('text',{{x:(L+W-R)/2,y:H-14,fill:'#a8aaa9','font-size':12,'text-anchor':'middle'}},'onset 群組順序（由慢到快測試的時間順序）'));
    svg.append(element('text',{{x:18,y:(T+H-B)/2,fill:'#a8aaa9','font-size':12,'text-anchor':'middle',transform:`rotate(-90 18 ${{(T+H-B)/2}})`}},'距離上一個 onset（ms）'));
  }}

  function drawTimeline() {{
    const svg=document.getElementById('timeline-chart'); clear(svg);
    const W=1100,H=370,L=126,R=26,T=28,B=42; const group=D.groups[selectedIndex]; const start=group.time_ms-localSpanMs, end=group.time_ms+localSpanMs;
    const zones=[...new Set(D.events.map(event=>event.zone_name))]; const x=value=>scale(value,start,end,L,W-R); const y=index=>T+index*((H-B-T)/(Math.max(zones.length-1,1)));
    const tickStep=localSpanMs<=1500?500:localSpanMs<=3000?1000:2000;
    for(let t=Math.ceil(start/tickStep)*tickStep;t<=end;t+=tickStep) {{ const xx=x(t); svg.append(element('line',{{x1:xx,y1:T-7,x2:xx,y2:H-B+7,stroke:'rgba(242,239,232,.09)','stroke-width':1}})); svg.append(element('text',{{x:xx,y:H-14,fill:'#a8aaa9','font-size':11,'text-anchor':'middle'}},`${{((t-group.time_ms)/1000).toFixed(1)}}s`)); }}
    zones.forEach((zone,index)=>{{ const yy=y(index); svg.append(element('line',{{x1:L,y1:yy,x2:W-R,y2:yy,stroke:'rgba(242,239,232,.13)','stroke-width':1}})); svg.append(element('text',{{x:L-12,y:yy+4,fill:color(zone),'font-size':13,'text-anchor':'end'}},zone)); }});
    const center=x(group.time_ms); svg.append(element('rect',{{x:center-1,y:T-10,width:2,height:H-B-T+20,fill:'#e0a106',opacity:.9}})); svg.append(element('text',{{x:center,y:17,fill:'#e0a106','font-size':12,'text-anchor':'middle'}},'選取 onset'));
    D.events.filter(event=>event.time_ms>=start&&event.time_ms<=end).forEach(event=>{{ const zoneIndex=zones.indexOf(event.zone_name); if(zoneIndex<0)return; const selected=event.group_id===group.group_id; const radius=selected?8:5; const dot=element('circle',{{cx:x(event.time_ms),cy:y(zoneIndex),r:radius,fill:color(event.zone_name),stroke:selected?'#f2efe8':'#0b0c0e','stroke-width':selected?2:1}}); dot.style.cursor='pointer'; dot.addEventListener('click',()=>selectGroup(event.group_id-1)); const title=element('title',{{}},`#${{event.event_id}}｜${{event.zone_name}}｜${{time(event.time_ms)}}｜velocity ${{event.velocity}}`); dot.append(title); svg.append(dot); }});
    svg.append(element('text',{{x:(L+W-R)/2,y:H-14,fill:'#a8aaa9','font-size':12,'text-anchor':'middle'}},`相對選取 onset 時間｜局部範圍 ±${{(localSpanMs/1000).toFixed(1)}} 秒`));
  }}

  function updateDetail() {{
    const group=D.groups[selectedIndex]; document.getElementById('selected-summary').textContent=`群組 #${{group.group_id}} · ${{time(group.time_ms)}} · ${{group.events.length}} 筆 MIDI`;
    document.getElementById('detail-group').textContent=`群組 #${{group.group_id}} · ${{time(group.time_ms)}}`;
    const events=document.getElementById('detail-events'); events.innerHTML=''; group.events.forEach(event=>{{ const chip=document.createElement('span'); chip.className='event-chip'; chip.style.background=color(event.zone_name); chip.innerHTML=`#${{event.event_id}} ${{event.zone_name}} <small>note ${{event.note}} · vel ${{event.velocity}}</small>`; events.append(chip); }});
    document.getElementById('detail-previous').textContent=fmt(group.previous_gap_ms); document.getElementById('detail-next').textContent=fmt(group.next_gap_ms);
    document.getElementById('detail-note').textContent=group.events.length>1 ? `本群組有 ${{group.events.length}} 筆 MIDI，起始時間差不超過 ${{S.simultaneous_tolerance_ms}} ms，依規則視為同時 onset，不當作連續手速。` : '本群組只有一筆 MIDI；前後間隔可用來判讀其與相鄰動作的時間距離。';
  }}

  function drawTable() {{
    const body=document.getElementById('fastest-table'); body.innerHTML=''; [...D.intervals].sort((a,b)=>a.gap_ms-b.gap_ms).slice(0,18).forEach(item=>{{ const group=D.groups[item.to_group_id-1]; const row=document.createElement('tr'); if(group.group_id===D.groups[selectedIndex].group_id) row.className='active'; row.innerHTML=`<td>#${{group.group_id}}</td><td>${{time(group.time_ms)}}</td><td>${{fmt(item.gap_ms)}}</td><td>${{notes(group.events)}}</td>`; row.addEventListener('click',()=>selectGroup(group.group_id-1)); body.append(row); }});
  }}

  function selectGroup(index) {{ selectedIndex=Math.max(0,Math.min(D.groups.length-1,index)); updateDetail(); drawIntervalChart(); drawTimeline(); drawTable(); }}
  document.querySelectorAll('[data-action]').forEach(button=>button.addEventListener('click',()=>{{ const action=button.dataset.action; if(action==='first')selectGroup(0); else if(action==='last')selectGroup(D.groups.length-1); else if(action==='previous')selectGroup(selectedIndex-1); else if(action==='next')selectGroup(selectedIndex+1); else if(action==='fastest')selectGroup((S.minimum_sequential_onset_gap?.to_group_id||1)-1); }}));
  document.getElementById('span-select').addEventListener('change',event=>{{ localSpanMs=Number(event.target.value); drawTimeline(); }});
  selectGroup(selectedIndex);
}})();
</script>
</body>
</html>"""


def main() -> int:
    if not MIDI_PATH.is_file():
        raise FileNotFoundError(f"Missing MIDI file: {MIDI_PATH}")
    summary, viewer_data = build_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(render_html(viewer_data), encoding="utf-8")
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(report_markdown(summary), encoding="utf-8")
    print(f"HTML: {OUTPUT_HTML}")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Report: {OUTPUT_REPORT}")
    print(f"Minimum raw adjacent gap: {fmt_ms(summary['minimum_raw_adjacent_gap']['gap_ms'] if summary['minimum_raw_adjacent_gap'] else None)}")
    print(f"Minimum sequential onset gap: {fmt_ms(summary['minimum_sequential_onset_gap']['gap_ms'] if summary['minimum_sequential_onset_gap'] else None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
