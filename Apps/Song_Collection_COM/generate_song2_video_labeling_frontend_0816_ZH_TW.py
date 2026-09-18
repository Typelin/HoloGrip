"""Generate the Song 2 video-assisted MIDI hand-labeling page.

The page is intentionally a review tool. It never edits Raw CSV, MIDI, or
pipeline outputs. Manual labels stay in browser storage until exported as CSV
or JSON by the reviewer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
sys.path.insert(0, str(APP_DIR))

import midi_label_pipeline as pipeline  # noqa: E402
import midi_quality_audit_0808_ZH_TW as audit  # noqa: E402


CSV_PATH = PROJECT_ROOT / "Data" / "Raw" / "Song_Collection_COM" / "S20260812_P01_song02_raw_100hz_20260812_112101.csv"
MIDI_PATH = Path(r"C:\Users\Typelin_Station\Downloads\0812T1127_去除雜訊T1213.mid")
OUTPUT_DIR = PROJECT_ROOT / "Data" / "Derived" / "Song_Collection_COM" / "S20260812_P01_song02_T1127_cleaned"
VIDEO_PATH = OUTPUT_DIR / "S20260812_P01_song02_T1127_video_h264.mp4"
OUTPUT_HTML = OUTPUT_DIR / "HoloGrip_Song2_影片人工標記_0816_ZH_TW.html"

BPM = 110.0
CSV_OFFSET_MS = 6_027
VIDEO_OFFSET_MS = 4_960
WINDOW_RADIUS_MS = 80
GROUP_TOLERANCE_MS = 12


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HoloGrip Song 2｜影片人工標記</title>
<style>
:root{--bg:#f4f6f8;--surface:#fff;--ink:#172433;--muted:#5d6a78;--line:#d6dde5;--blue:#226fba;--blue-soft:#e8f2fd;--red:#c94652;--red-soft:#fff0f2;--green:#318755;--green-soft:#e7f5ec;--amber:#a76714;--amber-soft:#fff4e3;--dark:#27313b;--shadow:0 8px 22px rgba(30,43,57,.08)}
*{box-sizing:border-box}html{background:var(--bg)}body{margin:0;color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI","Noto Sans TC",sans-serif}.app{max-width:1320px;margin:0 auto;padding:24px}.top{display:flex;justify-content:space-between;gap:20px;padding:0 0 18px;border-bottom:1px solid var(--line)}h1,h2,h3,p{margin:0}h1{font-size:25px;line-height:1.2}.eyebrow{color:var(--blue);font-size:12px;font-weight:700;letter-spacing:.08em}.sub{color:var(--muted);margin-top:6px}.status{align-self:start;background:var(--green-soft);color:#1e663d;border-radius:6px;padding:7px 10px;font-weight:650;font-size:13px;white-space:nowrap}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:18px 0}.stat{background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:12px 14px;box-shadow:var(--shadow)}.stat b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.stat span{display:block;color:var(--muted);font-size:12px;margin-top:2px}.section{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:18px;margin-top:16px;box-shadow:var(--shadow)}.section-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:14px}.section-head h2{font-size:18px}.section-head p{color:var(--muted);font-size:13px;margin-top:3px}.review-grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(330px,.9fr);gap:18px;align-items:start}.video-wrap{background:#111820;border-radius:6px;overflow:hidden;aspect-ratio:16/9}.video-wrap video{display:block;width:100%;height:100%;object-fit:contain}.control-row,.nav-row,.mark-row,.io-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}.button{border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink);padding:8px 11px;font:inherit;cursor:pointer}.button:hover{background:#f5f8fb;border-color:#aebdcb}.button:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid rgba(34,111,186,.28);outline-offset:2px}.primary{background:var(--blue);border-color:var(--blue);color:#fff}.primary:hover{background:#165c9e;border-color:#165c9e}.left-button{background:var(--blue-soft);border-color:#a8cbed;color:#145b9e}.right-button{background:var(--red-soft);border-color:#efb4bb;color:#a6323c}.dual-button{background:var(--green-soft);border-color:#a9d4b9;color:#236b43}.exclude-button{background:var(--amber-soft);border-color:#ebc887;color:#86530b}.button[disabled]{opacity:.5;cursor:not-allowed}.rate{color:var(--muted);font-variant-numeric:tabular-nums;margin-left:auto}.event-card{border:1px solid var(--line);border-radius:7px;padding:14px;background:#fbfcfd}.event-title{display:flex;align-items:center;justify-content:space-between;gap:10px}.event-title h3{font-size:18px}.badge{display:inline-flex;align-items:center;border-radius:999px;padding:4px 8px;font-size:12px;font-weight:700}.badge.pending{background:var(--amber-soft);color:#86530b}.badge.single{background:var(--blue-soft);color:#145b9e}.badge.dual{background:var(--green-soft);color:#236b43}.details{display:grid;grid-template-columns:auto minmax(0,1fr);gap:7px 12px;margin-top:13px;font-variant-numeric:tabular-nums}.details dt{color:var(--muted)}.details dd{margin:0;font-weight:600}.hint{color:var(--muted);font-size:13px;margin-top:12px}.activity{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:13px}.hand-score{padding:9px 10px;border-radius:6px;border:1px solid var(--line)}.hand-score b{display:block;font-size:18px;font-variant-numeric:tabular-nums}.hand-score.left{background:var(--blue-soft);border-color:#b8d4ef}.hand-score.right{background:var(--red-soft);border-color:#efc0c5}.hand-score span{display:block;font-size:12px;color:var(--muted)}.chart-wrap{margin-top:12px}.chart{width:100%;display:block;border:1px solid var(--line);border-radius:6px;background:#fff}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin-top:8px}.swatch{display:inline-block;width:20px;height:3px;border-radius:3px;vertical-align:middle;margin-right:5px}.blue{background:var(--blue)}.red{background:var(--red)}.green{background:var(--green)}.form-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;margin-top:13px}.form-row textarea{resize:vertical;min-height:42px;padding:8px;border:1px solid var(--line);border-radius:6px;font:inherit}.filter{display:grid;grid-template-columns:auto minmax(0,1fr);gap:10px;align-items:center;margin-top:4px}.filter label{color:var(--muted);font-size:13px}.filter select{width:100%;padding:8px;border:1px solid var(--line);border-radius:6px;background:#fff;font:inherit}.event-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-top:12px;max-height:300px;overflow:auto;padding-right:3px}.event-item{text-align:left;border:1px solid var(--line);border-radius:6px;background:#fff;padding:8px;cursor:pointer;font:inherit;color:var(--ink)}.event-item:hover{background:#f5f8fb}.event-item.selected{border-color:var(--blue);box-shadow:inset 3px 0 0 var(--blue);background:var(--blue-soft)}.event-item .meta{display:block;color:var(--muted);font-size:12px;margin-top:2px}.event-item .manual{display:block;font-size:12px;font-weight:700;margin-top:2px;color:#286a44}.notice{margin-top:12px;padding:10px 12px;background:#f7fafc;border-left:4px solid var(--blue);color:var(--muted);font-size:13px}.progress-line{display:flex;justify-content:space-between;gap:10px;color:var(--muted);font-size:13px;margin-top:10px}.progress-track{height:7px;background:#e7ecf1;border-radius:99px;overflow:hidden;margin-top:5px}.progress-track i{display:block;height:100%;background:var(--green);transition:width .15s}.footer-note{color:var(--muted);font-size:12px;margin:14px 0 0}.hidden{display:none!important}@media(max-width:880px){.review-grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.event-list{max-height:210px}}@media(max-width:540px){.app{padding:14px}.top{display:block}.status{display:inline-block;margin-top:10px}.stats{grid-template-columns:1fr 1fr}.event-list{grid-template-columns:1fr}.form-row{grid-template-columns:1fr}.rate{width:100%;margin-left:0}}
</style>
</head>
<body>
<main class="app">
  <header class="top">
    <div><div class="eyebrow">HOLOGRIP / SONG 2 / REVIEW</div><h1>影片輔助 MIDI 手別標記</h1><p class="sub">以 MIDI 中心點為基準，可單次或循環播放目前事件的 ±80 ms；CSV 折線採共同 song_time_ms。</p></div>
    <div class="status">已預載 Song 2 清理版</div>
  </header>
  <section class="stats" aria-label="資料品質摘要">
    <div class="stat"><b>311</b><span>清理後 MIDI 事件</span></div>
    <div class="stat"><b>0</b><span>±80 ms 異常事件</span></div>
    <div class="stat"><b>60</b><span>高信心單手：左 18 / 右 42</span></div>
    <div class="stat"><b>64 / 311</b><span>自動高信心，共 20.6%</span></div>
  </section>
  <section class="section">
    <div class="section-head"><div><h2>目前事件：影片、MIDI、CSV</h2><p>藍線是左手活動度，紅線是右手活動度。綠色範圍是本筆的正式訓練窗口 ±80 ms。</p></div><div id="selection-count" class="badge pending">待影片標記</div></div>
    <div class="active-event" aria-live="polite"><div><span>目前顯示</span><strong id="current-event-banner">載入中</strong><small id="current-event-times"></small></div><label>跳至 MIDI 編號 <input id="event-jump" type="number" min="1" max="311" step="1" inputmode="numeric"></label><button class="button" type="button" id="jump-event">切換</button></div>
    <div class="review-grid">
      <div>
        <div class="video-wrap"><video id="review-video" controls preload="auto" src="__VIDEO_SRC__">此瀏覽器無法播放影片。</video></div>
        <div class="control-row">
          <button class="button primary" type="button" id="play-window-once">播放一次目前 ±80 ms</button>
          <button class="button" type="button" id="play-window">循環播放目前 ±80 ms</button>
          <button class="button" type="button" id="pause-window">停止播放</button>
          <button class="button" type="button" id="seek-window">回到窗口起點</button>
          <button class="button" type="button" data-rate="1">1.0 倍</button>
          <button class="button" type="button" data-rate="0.5">0.5 倍</button>
          <button class="button" type="button" data-rate="0.25">0.25 倍</button>
          <button class="button" type="button" data-rate="0.125">0.125 倍</button>
          <span id="play-rate" class="rate">影片速度：1.0 倍</span>
        </div>
        <div class="chart-wrap"><svg id="activity-chart" class="chart" role="img" aria-label="目前 MIDI 事件前後的左右手活動度與其他 MIDI 事件"></svg></div>
        <div class="legend"><span><i class="swatch blue"></i>左手活動度</span><span><i class="swatch red"></i>右手活動度</span><span><i class="swatch green"></i>訓練窗口 ±80 ms</span><span>黑色中線：目前 MIDI</span><span>上方方塊：附近其他 MIDI</span></div>
      </div>
      <aside class="event-card" aria-live="polite">
        <div class="event-title"><h3 id="event-name">載入事件中</h3><span id="event-status" class="badge pending">待影片標記</span></div>
        <dl class="details"><dt>MIDI 編號</dt><dd id="event-id">-</dd><dt>MIDI 時間</dt><dd id="midi-time">-</dd><dt>CSV 中心</dt><dd id="csv-time">-</dd><dt>影片中心</dt><dd id="video-time">-</dd><dt>附近 MIDI</dt><dd id="nearby-midi">-</dd><dt>自動建議</dt><dd id="auto-suggestion">-</dd></dl>
        <div class="activity"><div class="hand-score left"><b id="left-peak">-</b><span>左手窗口峰值</span></div><div class="hand-score right"><b id="right-peak">-</b><span>右手窗口峰值</span></div></div>
        <p id="event-hint" class="hint"></p>
        <div class="mark-row">
          <button type="button" class="button left-button" data-mark="left">確認左手</button>
          <button type="button" class="button right-button" data-mark="right">確認右手</button>
          <button type="button" class="button dual-button" data-mark="dual">屬於雙手同時組</button>
          <button type="button" class="button exclude-button" data-mark="exclude">排除本筆</button>
          <button type="button" class="button" data-mark="uncertain">暫不確定</button>
          <button type="button" class="button" id="clear-mark">清除標記</button>
        </div>
        <div class="form-row"><textarea id="manual-note" aria-label="人工判讀備註" placeholder="可選：記錄影片中看到的手別、鼓位或不確定原因"></textarea><button class="button" type="button" id="save-note">儲存備註</button></div>
        <p class="hint">雙手同時組：請分別選取兩筆 MIDI，再以「確認左手／確認右手」完成兩筆的手別分配。「屬於雙手同時組」只保留群組關係，不能取代個別手別標記。</p>
      </aside>
    </div>
    <div class="nav-row">
      <button class="button" type="button" id="prev-event">上一筆 MIDI</button>
      <button class="button" type="button" id="next-event">下一筆 MIDI</button>
      <button class="button" type="button" id="next-review">下一筆待影片判讀</button>
      <button class="button" type="button" id="next-dual">下一個雙手 MIDI 組</button>
    </div>
  </section>
  <section class="section">
    <div class="section-head"><div><h2>事件導覽與標記匯出</h2><p>標記先保存在本機瀏覽器；完成後下載 CSV，作為後續訓練資料的人工真值輸入。</p></div></div>
    <div class="filter"><label for="event-filter">清單篩選</label><select id="event-filter"><option value="review">待影片判讀（251，含 4 筆雙手候選）</option><option value="all">全部 MIDI（311）</option><option value="single">自動高信心單手（60）</option><option value="dual">所有雙手同時 MIDI 組（14 筆 / 7 組）</option><option value="marked">已人工標記</option></select></div>
    <div id="event-list" class="event-list" aria-label="MIDI 事件清單"></div>
    <div class="progress-line"><span id="progress-text">已人工標記 0 / 311</span><span id="progress-breakdown">左 0｜右 0｜雙手組 0｜排除 0｜暫不確定 0</span></div>
    <div class="progress-track" role="progressbar" aria-label="人工標記進度" aria-valuemin="0" aria-valuemax="311" aria-valuenow="0"><i id="progress-fill"></i></div>
    <div class="io-row"><button type="button" class="button primary" id="download-csv">下載人工標記 CSV</button><button type="button" class="button" id="download-json">下載備份 JSON</button><label class="button" for="import-json">匯入 JSON 備份</label><input class="hidden" type="file" id="import-json" accept="application/json"></div>
    <div class="notice">輸出 CSV 會保留原始 MIDI note、鼓點名稱、MIDI/CSV/影片中心時間、自動建議與人工結果。原始 CSV、MIDI、影片與 pipeline 輸出都不會被這個頁面修改。</div>
    <p class="footer-note">對齊參數固定：MIDI → CSV +6,027 ms；MIDI → 影片 +4,960 ms；窗口 ±80 ms。影片為 60 FPS，單次窗口總長 160 ms，約 10 個影格。</p>
  </section>
</main>
<script>
const DATA=__DATA__;
const STORE_KEY='hologrip-song2-video-labels-0816-v1';
const video=document.getElementById('review-video');
const listNode=document.getElementById('event-list');
const filterNode=document.getElementById('event-filter');
const noteNode=document.getElementById('manual-note');
let index=0,playMode='stopped',rate=1,playheadFrame=0;
let labels={};
try{labels=JSON.parse(localStorage.getItem(STORE_KEY)||'{}')||{}}catch(_){labels={}}
const byId=new Map(DATA.events.map((event,position)=>[event.id,position]));
const fmtMs=value=>`${(Number(value)/1000).toFixed(3)} s`;
const fmtG=value=>`${Number(value).toFixed(2)} g`;
const current=()=>DATA.events[index];
const isMarked=event=>Boolean(labels[event.id]?.mark);
const isReview=event=>event.auto_status==='待人工判讀'||event.auto_status==='高信心雙手';
const currentMark=event=>labels[event.id]?.mark||'';
function typeLabel(mark){return({left:'已確認左手',right:'已確認右手',dual:'雙手同時組',exclude:'已排除',uncertain:'暫不確定'})[mark]||''}
function badgeClass(event){return event.auto_status==='高信心單手'?'single':event.auto_status==='高信心雙手'?'dual':'pending'}
function eventVideoCenter(event){return(event.midi_ms+DATA.video_offset_ms)/1000}
function eventSegment(event){const center=eventVideoCenter(event),radius=DATA.window_radius_ms/1000;return{start:Math.max(0,center-radius),end:center+radius,center}}
function nearby(event){return DATA.events.filter(other=>Math.abs(other.csv_ms-event.csv_ms)<=DATA.window_radius_ms)}
function setVideoTime(value){const seek=()=>{const duration=Number.isFinite(video.duration)?video.duration:value;video.currentTime=Math.max(0,Math.min(value,duration))};if(video.readyState>=1)seek();else video.addEventListener('loadedmetadata',seek,{once:true})}
function stopPlayback(){playMode='stopped';video.pause();cancelAnimationFrame(playheadFrame);updatePlayhead()}
function watchPlayback(){if(playMode==='stopped')return;const segment=eventSegment(current());if(video.currentTime>=segment.end-.003){if(playMode==='loop'){video.currentTime=segment.start;video.play().catch(()=>{playMode='stopped'})}else{playMode='stopped';video.currentTime=segment.end;video.pause();updatePlayhead();return}}updatePlayhead();playheadFrame=requestAnimationFrame(watchPlayback)}
function playWindow(mode){stopPlayback();const eventId=current().id,segment=eventSegment(current());playMode=mode;const begin=()=>{if(playMode!==mode||current().id!==eventId)return;video.playbackRate=rate;video.currentTime=segment.start;video.play().catch(()=>{playMode='stopped'});cancelAnimationFrame(playheadFrame);playheadFrame=requestAnimationFrame(watchPlayback)};if(video.readyState>=1)begin();else video.addEventListener('loadedmetadata',begin,{once:true})}
function setRate(value){rate=value;video.playbackRate=rate;document.getElementById('play-rate').textContent=`影片速度：${rate} 倍`;document.querySelectorAll('[data-rate]').forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.rate)===rate)))}
function updatePlayhead(){const line=document.getElementById('chart-playhead');if(!line)return;const event=current();const relative=video.currentTime*1000-DATA.video_offset_ms-event.midi_ms;const look=DATA.lookaround_ms;const x=50+((relative+look)/(look*2))*910;line.setAttribute('x1',String(Math.max(50,Math.min(960,x))));line.setAttribute('x2',String(Math.max(50,Math.min(960,x))))}
function chart(event){const look=DATA.lookaround_ms,start=event.csv_ms-look,end=event.csv_ms+look;const points=DATA.energy.filter(point=>point.t>=start&&point.t<=end);const maximum=Math.max(1,...points.map(point=>Math.max(point.l,point.r)));const x=time=>50+((time-start)/(look*2))*910;const y=value=>238-(value/maximum)*178;const poly=key=>points.map(point=>`${x(point.t).toFixed(1)},${y(point[key]).toFixed(1)}`).join(' ');const markers=DATA.events.filter(other=>Math.abs(other.csv_ms-event.csv_ms)<=look).map(other=>{const px=x(other.csv_ms),selected=other.id===event.id;return `<rect x="${(px-3).toFixed(1)}" y="18" width="6" height="14" rx="1" fill="${selected?'#27313b':'#a76714'}"><title>${selected?'目前':'附近'} MIDI #${other.id}｜${other.name}｜${fmtMs(other.midi_ms)}</title></rect>`}).join('');const windowLeft=x(event.csv_ms-DATA.window_radius_ms),windowWidth=x(event.csv_ms+DATA.window_radius_ms)-windowLeft;return `<svg id="activity-chart" class="chart" role="img" aria-label="目前事件前後活動度" viewBox="0 0 1010 270"><rect x="50" y="20" width="910" height="218" fill="#ffffff"/><rect x="${windowLeft.toFixed(1)}" y="20" width="${windowWidth.toFixed(1)}" height="218" fill="#e7f5ec"/><path d="M50 238H960M50 149H960M50 60H960" stroke="#d6dde5" stroke-width="1"/><text x="12" y="64" fill="#5d6a78" font-size="12">${maximum.toFixed(1)} g</text><text x="20" y="242" fill="#5d6a78" font-size="12">0 g</text>${markers}<polyline points="${poly('l')}" fill="none" stroke="#226fba" stroke-width="2.4"/><polyline points="${poly('r')}" fill="none" stroke="#c94652" stroke-width="2.4"/><line x1="505" x2="505" y1="20" y2="238" stroke="#27313b" stroke-width="2"/><line id="chart-playhead" x1="505" x2="505" y1="20" y2="238" stroke="#318755" stroke-width="2" stroke-dasharray="4 3"/><text x="50" y="260" fill="#5d6a78" font-size="12">−${look} ms</text><text x="485" y="260" fill="#27313b" font-size="12">MIDI</text><text x="925" y="260" fill="#5d6a78" font-size="12">+${look} ms</text></svg>`}
function filteredEvents(){const mode=filterNode.value;if(mode==='all')return DATA.events;if(mode==='review')return DATA.events.filter(isReview);if(mode==='single')return DATA.events.filter(event=>event.auto_status==='高信心單手');if(mode==='dual')return DATA.events.filter(event=>event.group_event_count===2);return DATA.events.filter(isMarked)}
function renderList(){const events=filteredEvents();listNode.innerHTML=events.length?events.map(event=>{const chosen=event.id===current().id?' selected':'';const manual=typeLabel(currentMark(event));return `<button class="event-item${chosen}" type="button" data-event-id="${event.id}"><b>#${event.id}｜${event.name}</b><span class="meta">${fmtMs(event.midi_ms)}｜${event.auto_status}</span>${manual?`<span class="manual">${manual}</span>`:''}</button>`}).join(''):'<p class="hint">此篩選目前沒有事件。</p>';listNode.querySelectorAll('[data-event-id]').forEach(button=>button.addEventListener('click',()=>selectById(Number(button.dataset.eventId),true)))}
function renderProgress(){const values=Object.values(labels).filter(item=>item?.mark);const count=values.length,counts={left:0,right:0,dual:0,exclude:0,uncertain:0};values.forEach(item=>{if(counts[item.mark]!==undefined)counts[item.mark]+=1});document.getElementById('progress-text').textContent=`已人工標記 ${count} / ${DATA.events.length}`;document.getElementById('progress-breakdown').textContent=`左 ${counts.left}｜右 ${counts.right}｜雙手組 ${counts.dual}｜排除 ${counts.exclude}｜暫不確定 ${counts.uncertain}`;const fill=document.getElementById('progress-fill');fill.style.width=`${(count/DATA.events.length)*100}%`;fill.parentElement.setAttribute('aria-valuenow',String(count))}
function render(){const event=current(),mark=currentMark(event),isDual=event.group_event_count===2;document.getElementById('event-name').textContent=`#${event.id}｜${event.name}（note ${event.note}）`;const status=document.getElementById('event-status');status.className=`badge ${mark?'dual':badgeClass(event)}`;status.textContent=mark?typeLabel(mark):event.auto_status;document.getElementById('selection-count').className=`badge ${badgeClass(event)}`;document.getElementById('selection-count').textContent=`目前第 ${index+1} / ${DATA.events.length} 筆`;document.getElementById('event-id').textContent=`${event.id}｜velocity ${event.velocity}`;document.getElementById('midi-time').textContent=fmtMs(event.midi_ms);document.getElementById('csv-time').textContent=fmtMs(event.csv_ms);document.getElementById('video-time').textContent=fmtMs(eventVideoCenter(event)*1000);const close=nearby(event);document.getElementById('nearby-midi').textContent=close.map(other=>`#${other.id} ${other.name}`).join('、');document.getElementById('auto-suggestion').textContent=`${event.auto_hand==='L'?'左手':'右手'} ${Math.round(event.auto_confidence*100)}%（僅供判讀）`;document.getElementById('left-peak').textContent=fmtG(event.left_peak);document.getElementById('right-peak').textContent=fmtG(event.right_peak);document.getElementById('event-hint').textContent=isDual?'這是兩筆幾乎同時的不同 MIDI。先看影片，再分別為每一筆指定左手或右手。':event.auto_status==='高信心單手'?'此筆已通過自動高信心規則，但仍可用影片抽查確認。':'此筆未通過單手自動規則；請以影片判斷是否可保留為訓練標籤。';noteNode.value=labels[event.id]?.note||'';document.getElementById('activity-chart').outerHTML=chart(event);renderList();renderProgress();setVideoTime(eventVideoCenter(event));}
function selectById(id,seek){const next=byId.get(id);if(next===undefined)return;stopPlayback();index=next;render();if(!seek)return}
function step(delta){selectById(DATA.events[(index+delta+DATA.events.length)%DATA.events.length].id,true)}
function nextMatch(predicate){for(let offset=1;offset<=DATA.events.length;offset+=1){const event=DATA.events[(index+offset)%DATA.events.length];if(predicate(event)){selectById(event.id,true);return}}}
function persist(){localStorage.setItem(STORE_KEY,JSON.stringify(labels));renderList();renderProgress()}
function markCurrent(mark){const event=current();labels[event.id]={...(labels[event.id]||{}),mark,note:noteNode.value.trim(),updated_at:new Date().toISOString()};persist();render()}
function csvEscape(value){const raw=String(value??'');return /[",\n]/.test(raw)?`"${raw.replaceAll('"','""')}"`:raw}
function download(name,type,content){const blob=new Blob([content],{type});const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),0)}
function downloadCsv(){const header=['event_id','midi_time_ms','csv_center_ms','video_center_ms','midi_note','drum','velocity','auto_status','auto_hand','auto_confidence','manual_label','manual_note','updated_at'];const rows=DATA.events.map(event=>{const mark=labels[event.id]||{};return[event.id,event.midi_ms,event.csv_ms,(eventVideoCenter(event)*1000).toFixed(3),event.note,event.name,event.velocity,event.auto_status,event.auto_hand,event.auto_confidence,mark.mark||'',mark.note||'',mark.updated_at||'']});download('Song2_人工影片標記_0816.csv','text/csv;charset=utf-8',[header,...rows].map(row=>row.map(csvEscape).join(',')).join('\r\n'))}
function importBackup(file){const reader=new FileReader();reader.onload=()=>{try{const input=JSON.parse(String(reader.result));labels=input.labels||input||{};persist();render()}catch(_){window.alert('無法讀取這個 JSON 備份。')}};reader.readAsText(file,'utf-8')}
document.getElementById('play-window-once').addEventListener('click',()=>playWindow('once'));document.getElementById('play-window').addEventListener('click',()=>playWindow('loop'));document.getElementById('pause-window').addEventListener('click',stopPlayback);document.getElementById('seek-window').addEventListener('click',()=>{stopPlayback();setVideoTime(eventSegment(current()).start)});document.querySelectorAll('[data-rate]').forEach(button=>button.addEventListener('click',()=>setRate(Number(button.dataset.rate))));document.getElementById('prev-event').addEventListener('click',()=>step(-1));document.getElementById('next-event').addEventListener('click',()=>step(1));document.getElementById('next-review').addEventListener('click',()=>nextMatch(isReview));document.getElementById('next-dual').addEventListener('click',()=>nextMatch(event=>event.group_event_count===2));document.querySelectorAll('[data-mark]').forEach(button=>button.addEventListener('click',()=>markCurrent(button.dataset.mark)));document.getElementById('clear-mark').addEventListener('click',()=>{delete labels[current().id];persist();render()});document.getElementById('save-note').addEventListener('click',()=>{const event=current();labels[event.id]={...(labels[event.id]||{}),note:noteNode.value.trim(),updated_at:new Date().toISOString()};persist()});filterNode.addEventListener('change',renderList);document.getElementById('download-csv').addEventListener('click',downloadCsv);document.getElementById('download-json').addEventListener('click',()=>download('Song2_人工影片標記_0816.json','application/json;charset=utf-8',JSON.stringify({schema_version:1,source:DATA.source,labels},null,2)));document.getElementById('import-json').addEventListener('change',event=>{const file=event.target.files?.[0];if(file)importBackup(file);event.target.value=''});video.addEventListener('seeking',updatePlayhead);video.addEventListener('timeupdate',updatePlayhead);video.addEventListener('pause',()=>{if(playMode!=='stopped'){playMode='stopped';cancelAnimationFrame(playheadFrame);updatePlayhead()}});setRate(1);render();
</script>
<style>
.active-event{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:0 0 14px;padding:10px 12px;border:1px solid #b8d4ef;border-left:4px solid #226fba;border-radius:6px;background:#f5f9fd;color:#172433}.active-event span{display:block;color:#5d6a78;font-size:12px}.active-event strong{display:block;font-size:17px;font-variant-numeric:tabular-nums}.active-event small{display:block;color:#5d6a78;margin-top:2px;font-variant-numeric:tabular-nums}.active-event label{display:flex;align-items:center;gap:7px;margin-left:auto;color:#5d6a78;font-size:13px}.active-event input{width:72px;padding:7px;border:1px solid #d6dde5;border-radius:6px;background:#fff;font:inherit;color:#172433}@media(max-width:540px){.active-event label{margin-left:0}}
</style>
<script>
(() => {
  const banner = document.getElementById('current-event-banner');
  const timing = document.getElementById('current-event-times');
  const input = document.getElementById('event-jump');
  const jump = document.getElementById('jump-event');
  const sync = () => {
    const name = document.getElementById('event-name')?.textContent || '';
    const id = (document.getElementById('event-id')?.textContent || '').split('｜')[0].trim();
    const midi = document.getElementById('midi-time')?.textContent || '-';
    const csv = document.getElementById('csv-time')?.textContent || '-';
    const video = document.getElementById('video-time')?.textContent || '-';
    banner.textContent = name;
    timing.textContent = `MIDI ${midi} ｜ CSV ${csv} ｜ 影片 ${video}`;
    input.value = id;
    const chart = document.getElementById('activity-chart');
    const old = document.getElementById('chart-event-label');
    if (chart && !old) {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.id = 'chart-event-label';
      label.setAttribute('x', '62');
      label.setAttribute('y', '15');
      label.setAttribute('fill', '#172433');
      label.setAttribute('font-size', '12');
      label.setAttribute('font-weight', '700');
      label.textContent = `目前事件：${name}`;
      chart.appendChild(label);
    }
  };
  const select = () => {
    const id = Number(input.value);
    if (Number.isInteger(id) && id >= 1 && id <= 311 && typeof window.selectById === 'function') window.selectById(id, true);
  };
  jump.addEventListener('click', select);
  input.addEventListener('keydown', event => { if (event.key === 'Enter') select(); });
  new MutationObserver(sync).observe(document.getElementById('event-name'), {childList:true, characterData:true, subtree:true});
  sync();
})();
</script>
</body>
</html>
"""


def build_data() -> dict[str, object]:
    samples, raw_info = pipeline.read_raw_csv(CSV_PATH)
    energy = audit.energy_rows(samples, raw_info["duration_ms"])
    parsed = pipeline.parse_midi(MIDI_PATH, BPM)
    audit_events = []
    for event in parsed["events"]:
        # Match the production quality-audit rule exactly: hand score is the
        # maximum activity inside this page's ±80 ms training window.
        points = audit.window_rows(energy, event.time_ms + CSV_OFFSET_MS, WINDOW_RADIUS_MS)
        left_peak = max((point["l"] for point in points), default=0.0)
        right_peak = max((point["r"] for point in points), default=0.0)
        total = left_peak + right_peak
        hand_info = {
            "hand": "L" if left_peak > right_peak else "R" if total else "?",
            "confidence": max(left_peak, right_peak) / total if total else 0.0,
            "left_peak": left_peak,
            "right_peak": right_peak,
        }
        audit_events.append(audit.make_event_row(event, hand_info))
    summary, reviewed, _ = audit.audit_radius(
        audit_events,
        energy,
        WINDOW_RADIUS_MS,
        CSV_OFFSET_MS,
        duplicate_window_ms=20,
        group_tolerance_ms=GROUP_TOLERANCE_MS,
    )
    events = []
    for row in reviewed:
        events.append({
            "id": int(row["event_id"]),
            "midi_ms": float(row["midi_time_ms"]),
            "csv_ms": round(float(row["midi_time_ms"]) + CSV_OFFSET_MS, 3),
            "note": int(row["midi_note"]),
            "name": row["zone_name"],
            "velocity": int(row["velocity"]),
            "auto_hand": row["hand_candidate"],
            "auto_confidence": float(row["hand_confidence"]),
            "left_peak": float(row["left_peak_g"]),
            "right_peak": float(row["right_peak_g"]),
            "auto_status": row["label_status"],
            "group_event_count": sum(item["group_index"] == row["group_index"] for item in reviewed),
        })
    if summary["total_midi_events"] != 311:
        raise RuntimeError(f"Unexpected Song 2 MIDI count: {summary['total_midi_events']}")
    if summary["high_confidence_total_midi_events"] != 64:
        raise RuntimeError(
            "Song 2 quality statistics changed; expected 64 automatic high-confidence events, "
            f"got {summary['high_confidence_total_midi_events']}"
        )
    return {
        "source": {
            "midi": str(MIDI_PATH),
            "csv": str(CSV_PATH),
            "video": str(VIDEO_PATH),
        },
        "window_radius_ms": WINDOW_RADIUS_MS,
        "csv_offset_ms": CSV_OFFSET_MS,
        "video_offset_ms": VIDEO_OFFSET_MS,
        "lookaround_ms": 240,
        "summary": summary,
        "events": events,
        "energy": energy,
    }


def main() -> int:
    required = (CSV_PATH, MIDI_PATH, VIDEO_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input:\n" + "\n".join(missing))
    data = build_data()
    html = HTML_TEMPLATE.replace("__VIDEO_SRC__", f"./{VIDEO_PATH.name}")
    html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"frontend={OUTPUT_HTML}")
    print(json.dumps(data["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
