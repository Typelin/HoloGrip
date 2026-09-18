"""Replay Song 2 Raw CSV vs labeled GT. Playable product demo with live rates."""
from __future__ import annotations

import json
import math
import os
import sys
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from product_hit_and_zone import (
    CLEANED_DIR,
    EVAL_JSON_PATH,
    MODEL_PATH,
    classify_detections,
    evaluate,
    load_gt,
    load_raw_rows,
    load_zone_model,
    replay_hits,
    rows_to_samples,
    write_eval_json,
)

DEMO_HTML = os.path.join(CLEANED_DIR, "HoloGrip_Song2_產品Demo_即時正確率_0821_ZH_TW.html")
HAND_ZH = {"L": "左手", "R": "右手"}
DRUM_COLOR = {
    "小鼓": "#00f0ff",
    "高音 Tom": "#38bdf8",
    "中音 Tom": "#818cf8",
    "落地 Tom": "#a855f7",
    "Hi-Hat": "#eab308",
    "Crash": "#f97316",
    "Ride": "#10b981",
}
STATUS_ZH = {"ok": "正確", "wrong": "鼓位錯", "extra": "多偵測", "miss": "漏打"}


def energy_grid(samples, step_ms=20):
    dur = max(samples["L"][-1]["song_time_ms"], samples["R"][-1]["song_time_ms"])
    n = int(math.ceil(dur / step_ms)) + 1
    left = [0.0] * n
    right = [0.0] * n
    for hand, key in (("L", left), ("R", right)):
        for s in samples[hand]:
            i = int(s["song_time_ms"] / step_ms)
            if 0 <= i < n:
                v = abs(s["mag"] - 1.0)
                if v > key[i]:
                    key[i] = v
    return [{"t": i * step_ms, "l": round(left[i], 4), "r": round(right[i], 4)} for i in range(n)]


def build_events(hits, report):
    matched = {m["pred_id"]: m for m in report["matched_rows"]}
    events = []
    for h in hits:
        m = matched.get(h["pred_id"])
        if m is None:
            status = "extra"
            gt_hand = gt_drum = None
            dt = None
        else:
            status = "ok" if m["zone_ok"] else "wrong"
            gt_hand, gt_drum, dt = m["hand"], m["drum"], round(m["dt_ms"], 1)
        events.append({
            "kind": "pred",
            "id": h["pred_id"],
            "t": h["trigger_ms"],
            "hand": h["hand"],
            "hand_zh": HAND_ZH[h["hand"]],
            "drum": h["pred_drum"],
            "color": DRUM_COLOR.get(h["pred_drum"], "#e2e8f0"),
            "mag": h["peak_accel_g"],
            "p": h["pred_proba"],
            "status": status,
            "status_zh": STATUS_ZH[status],
            "gt_hand": gt_hand,
            "gt_hand_zh": HAND_ZH.get(gt_hand) if gt_hand else None,
            "gt_drum": gt_drum,
            "dt_ms": dt,
        })
    for g in report["miss_rows"]:
        events.append({
            "kind": "miss",
            "id": f"m{g['event_id']}",
            "t": g["csv_center_ms"],
            "hand": g["hand"],
            "hand_zh": HAND_ZH[g["hand"]],
            "drum": None,
            "color": "#f87171",
            "mag": None,
            "p": None,
            "status": "miss",
            "status_zh": "漏打",
            "gt_hand": g["hand"],
            "gt_hand_zh": HAND_ZH[g["hand"]],
            "gt_drum": g["drum"],
            "dt_ms": None,
        })
    events.sort(key=lambda e: (e["t"], 0 if e["kind"] == "pred" else 1))
    return events


HEAD = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>HoloGrip 產品 Demo · 手+鼓 93.5%</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#070b16;color:#e2e8f0;font-family:"Microsoft JhengHei UI",sans-serif}
.top{padding:16px 24px 4px;display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap}
h1{margin:0;font-size:22px}
.sub{color:#94a3b8;font-size:13px;margin-top:4px}
.phase{padding:8px 14px;border-radius:999px;font-weight:700;font-size:13px;background:#1e293b;color:#94a3b8;white-space:nowrap}
.phase.live{background:#1e3a5f;color:#7dd3fc}
.phase.done{background:#14532d;color:#86efac}
.cards{display:flex;gap:10px;flex-wrap:wrap;margin:10px 24px}
.card{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:12px 14px;min-width:140px}
.card b{display:block;font-size:28px;font-variant-numeric:tabular-nums}
.card span{color:#94a3b8;font-size:12px;line-height:1.35}
.ok{color:#86efac}.wrong{color:#fbbf24}.extra{color:#94a3b8}.miss{color:#f87171}.teal{color:#67e8f9}
.end{display:none;margin:0 24px 10px;background:#052e16;border:1px solid #166534;border-radius:14px;padding:14px 16px}
.end.show{display:block}
.end h2{margin:0 0 8px;font-size:16px;color:#86efac}
.end grid,.end .g{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px 18px;font-size:14px}
.end b{font-variant-numeric:tabular-nums;color:#f8fafc}
.stage{margin:8px 24px;background:#0f172a;border:1px solid #1e293b;border-radius:16px;padding:20px;display:grid;grid-template-columns:1fr 1fr;gap:12px}
.box{text-align:center;padding:8px}
.lab{font-size:12px;color:#64748b;letter-spacing:1px}
.hand{font-size:18px;margin-top:4px}
.drum{font-size:36px;font-weight:800;margin:4px 0}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;margin-top:6px}
.badge.ok{background:#14532d;color:#86efac}
.badge.wrong{background:#713f12;color:#fde68a}
.badge.extra{background:#1e293b;color:#cbd5e1}
.badge.miss{background:#7f1d1d;color:#fecaca}
.bar{margin:0 24px 10px;background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:10px 14px}
#cv{width:100%;height:120px;display:block;cursor:pointer}
.ctrl{display:flex;gap:8px;align-items:center;margin:0 24px 10px;flex-wrap:wrap}
button,.ss{background:#1e293b;border:1px solid #334155;color:#f8fafc;padding:7px 11px;border-radius:8px;cursor:pointer;font-size:13px}
button.p{background:#2563eb;border-color:#2563eb}
button.on{background:#334155}
.time{font-variant-numeric:tabular-nums;color:#38bdf8;font-weight:700}
.log{margin:0 24px 28px;height:280px;overflow:auto;background:#0f172a;border:1px solid #1e293b;border-radius:12px}
.row{display:grid;grid-template-columns:88px 70px 1.1fr 1.1fr 72px 56px;gap:8px;padding:7px 12px;border-bottom:1px solid #1e2937;font-size:13px;align-items:center}
.row.head{color:#64748b;position:sticky;top:0;background:#0f172a;z-index:1}
.row.cur{outline:1px solid #38bdf8}
.L{color:#60a5fa}.R{color:#fb7185}
</style></head><body>
<div class="top">
  <div>
    <h1>產品 Demo：有打，且打哪一顆</h1>
    <div class="sub">Raw 100Hz → HitDetector → ±80ms IMU → 7 類鼓位。MIDI 只對答案，不進模型。手別來自手套 HAND_ID。按播放看即時正確率，播完定格全曲成績。</div>
  </div>
  <div class="phase" id="phase">尚未開始</div>
</div>
<div class="cards" id="cards"></div>
<div class="end" id="endbox"></div>
<div class="stage">
  <div class="box">
    <div class="lab">系統輸出（手套 IMU）</div>
    <div class="hand" id="sysHand">—</div>
    <div class="drum" id="sysDrum">等待播放</div>
  </div>
  <div class="box">
    <div class="lab">標記真值（只拿來對答案）</div>
    <div class="hand" id="gtHand">—</div>
    <div class="drum" id="gtDrum">—</div>
    <div class="badge extra" id="badge">尚未比對</div>
  </div>
</div>
<div class="bar"><canvas id="cv"></canvas></div>
<div class="ctrl">
  <button class="p" id="play">播放</button>
  <button id="back">重來</button>
  <button id="toend">跳到結束</button>
  <span>速度</span>
  <select class="ss" id="spd"><option>1</option><option>4</option><option selected>8</option><option>16</option></select>
  <span class="time" id="clock">00:00.000</span>
  <button data-f="all" class="on">全部</button>
  <button data-f="ok">正確</button>
  <button data-f="wrong">鼓位錯</button>
  <button data-f="extra">多偵測</button>
  <button data-f="miss">漏打</button>
</div>
<div class="log">
  <div class="row head"><span>時間</span><span>結果</span><span>系統</span><span>真值</span><span>g</span><span>差</span></div>
  <div id="log"></div>
</div>
<script>
const D = """

TAIL = r""";
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
let playing=false, t=0, last=0, idx=0, spd=8, filter='all';
function fmt(ms){const s=Math.max(0,ms/1000);const m=Math.floor(s/60);const r=s-m*60;return String(m).padStart(2,'0')+':'+r.toFixed(3).padStart(6,'0')}
function pct(x){return (x*100).toFixed(1)+'%'}
function atEnd(){return t>=D.duration_ms-0.5}
function tally(){
  let ok=0,wrong=0,extra=0,miss=0;
  for(let i=0;i<idx;i++){
    const s=D.events[i].status;
    if(s==='ok')ok++; else if(s==='wrong')wrong++; else if(s==='extra')extra++; else if(s==='miss')miss++;
  }
  const gt=ok+wrong+miss, pred=ok+wrong+extra, matched=ok+wrong;
  return {ok,wrong,extra,miss,gt,pred,matched};
}
function renderCards(){
  const S=D.stats, done=atEnd(), x=tally();
  const product=done?S.product_recall:(x.gt?x.ok/x.gt:0);
  const rec=done?S.hit_recall:(x.gt?x.matched/x.gt:0);
  const prec=done?S.hit_precision:(x.pred?x.matched/x.pred:0);
  const zone=done?S.zone_acc_given_hit:(x.matched?x.ok/x.matched:0);
  const ok=done?S.joint_ok:x.ok;
  const wrong=done?S.zone_wrong:x.wrong;
  const extra=done?S.extras:x.extra;
  const miss=done?S.misses:x.miss;
  const gtDen=done?S.gt_single:x.gt;
  const predDen=done?S.predictions:x.pred;
  const matchDen=done?S.matched:x.matched;
  document.getElementById('cards').innerHTML = `
    <div class="card"><b class="ok">${pct(product)}</b><span>手+鼓都對<br>${ok} / ${gtDen||0} 真值</span></div>
    <div class="card"><b class="teal">${pct(rec)}</b><span>打擊召回<br>有打到 ${matchDen} / ${gtDen||0}</span></div>
    <div class="card"><b>${pct(prec)}</b><span>打擊精確率<br>對到真值 / 系統 ${predDen||0}</span></div>
    <div class="card"><b>${pct(zone)}</b><span>偵測後鼓位對<br>${ok} / ${matchDen||0}</span></div>
    <div class="card"><b class="ok">${ok}</b><span>正確</span></div>
    <div class="card"><b class="wrong">${wrong}</b><span>鼓位錯</span></div>
    <div class="card"><b class="extra">${extra}</b><span>多偵測</span></div>
    <div class="card"><b class="miss">${miss}</b><span>漏打</span></div>`;
  const ph=document.getElementById('phase');
  const box=document.getElementById('endbox');
  if(!idx && t<=0){ ph.className='phase'; ph.textContent='尚未開始 · 按播放'; box.className='end'; box.innerHTML=''; return; }
  if(done){
    ph.className='phase done'; ph.textContent='本曲結束 · 手+鼓 93.5%';
    box.className='end show';
    box.innerHTML=`<h2>全曲定格（Song 2 · 單手真值 ${S.gt_single}）</h2>
      <div class="g">
        <div>手+鼓都對 <b>${pct(S.product_recall)}</b>　${S.joint_ok} / ${S.gt_single}</div>
        <div>打擊召回 <b>${pct(S.hit_recall)}</b>　${S.matched} / ${S.gt_single}</div>
        <div>打擊精確率 <b>${pct(S.hit_precision)}</b>　${S.matched} / ${S.predictions}</div>
        <div>偵測後鼓位 <b>${pct(S.zone_acc_given_hit)}</b>　${S.joint_ok} / ${S.matched}</div>
        <div>鼓位錯 <b>${S.zone_wrong}</b></div>
        <div>多偵測 <b>${S.extras}</b></div>
        <div>漏打 <b>${S.misses}</b></div>
        <div>系統輸出 <b>${S.predictions}</b></div>
      </div>`;
  } else {
    ph.className='phase live'; ph.textContent='播放中 · 即時累計（分母=目前已出現的真值）';
    box.className='end'; box.innerHTML='';
  }
}
function resize(){const r=cv.getBoundingClientRect();cv.width=r.width*devicePixelRatio;cv.height=120*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);draw()}
function xof(ms){return (ms/D.duration_ms)*(cv.getBoundingClientRect().width)}
function draw(){
  const w=cv.getBoundingClientRect().width,h=120;
  ctx.clearRect(0,0,w,h); ctx.fillStyle='#020617'; ctx.fillRect(0,0,w,h);
  const maxv=Math.max(0.2,...D.wave.map(p=>Math.max(p.l,p.r)));
  ctx.beginPath(); ctx.strokeStyle='#3b82f6'; ctx.lineWidth=1.3;
  D.wave.forEach((p,i)=>{const x=xof(p.t), y=h*0.46-p.l/maxv*h*0.38; i?ctx.lineTo(x,y):ctx.moveTo(x,y)}); ctx.stroke();
  ctx.beginPath(); ctx.strokeStyle='#f43f5e';
  D.wave.forEach((p,i)=>{const x=xof(p.t), y=h*0.54+p.r/maxv*h*0.38; i?ctx.lineTo(x,y):ctx.moveTo(x,y)}); ctx.stroke();
  D.events.forEach(e=>{
    const col=e.status==='ok'?'#22c55e':e.status==='wrong'?'#fbbf24':e.status==='miss'?'#ef4444':'#64748b';
    ctx.fillStyle=e.t<=t?col:'#1e293b';
    ctx.fillRect(xof(e.t), e.status==='miss'?h-10:4, 2, 8);
  });
  ctx.strokeStyle='#f8fafc'; ctx.beginPath(); ctx.moveTo(xof(t),0); ctx.lineTo(xof(t),h); ctx.stroke();
}
function paint(e){
  const sysH=document.getElementById('sysHand'), sysD=document.getElementById('sysDrum');
  const gtH=document.getElementById('gtHand'), gtD=document.getElementById('gtDrum'), b=document.getElementById('badge');
  if(e.kind==='pred'){
    sysH.textContent=e.hand_zh; sysH.className='hand '+e.hand;
    sysD.textContent=e.drum; sysD.style.color=e.color;
  } else {
    sysH.textContent='沒偵測到'; sysH.className='hand miss';
    sysD.textContent='—'; sysD.style.color='#f87171';
  }
  gtH.textContent=e.gt_hand_zh||'沒有對應真值'; gtH.className='hand '+(e.gt_hand||'');
  gtD.textContent=e.gt_drum||'—';
  b.className='badge '+e.status; b.textContent=e.status_zh;
}
function addRow(e){
  if(filter!=='all' && e.status!==filter) return;
  const row=document.createElement('div');
  row.className='row cur '+e.status; row.id='r'+e.id;
  const sys=e.kind==='pred'?(e.hand_zh+' '+e.drum):'—';
  const gt=e.gt_hand_zh? (e.gt_hand_zh+' '+e.gt_drum):'—';
  const mag=e.mag==null?'—':e.mag.toFixed(2)+'g';
  const dt=e.dt_ms==null?'—':Math.round(e.dt_ms)+'ms';
  row.innerHTML=`<span>${fmt(e.t)}</span><span class="${e.status}">${e.status_zh}</span><span class="${e.hand||''}">${sys}</span><span>${gt}</span><span>${mag}</span><span>${dt}</span>`;
  const log=document.getElementById('log');
  [...log.children].forEach(n=>n.classList.remove('cur'));
  log.prepend(row);
}
function rebuildLog(){
  document.getElementById('log').innerHTML='';
  D.events.filter(e=>e.t<=t && (filter==='all'||e.status===filter)).forEach(addRow);
}
function syncTo(ms){
  t=Math.max(0,Math.min(D.duration_ms,ms));
  idx=0;
  while(idx<D.events.length && D.events[idx].t<=t){ idx++; }
  if(idx>0) paint(D.events[idx-1]);
  rebuildLog();
  document.getElementById('clock').textContent=fmt(t);
  renderCards(); draw();
}
function tick(ts){
  if(playing){
    if(!last) last=ts;
    t += (ts-last)*spd; last=ts;
    if(t>D.duration_ms){ t=D.duration_ms; playing=false; document.getElementById('play').textContent='播放'; }
    while(idx<D.events.length && D.events[idx].t<=t){ paint(D.events[idx]); addRow(D.events[idx]); idx++; }
    document.getElementById('clock').textContent=fmt(t);
    renderCards(); draw();
  }
  requestAnimationFrame(tick);
}
function reset(){
  t=0; idx=0; last=0; playing=false;
  document.getElementById('play').textContent='播放';
  document.getElementById('log').innerHTML='';
  document.getElementById('sysHand').textContent='—';
  document.getElementById('sysDrum').textContent='等待播放';
  document.getElementById('gtHand').textContent='—';
  document.getElementById('gtDrum').textContent='—';
  document.getElementById('badge').className='badge extra';
  document.getElementById('badge').textContent='尚未比對';
  document.getElementById('clock').textContent=fmt(0);
  renderCards(); draw();
}
document.getElementById('play').onclick=()=>{playing=!playing; last=0; document.getElementById('play').textContent=playing?'暫停':'播放'};
document.getElementById('back').onclick=reset;
document.getElementById('toend').onclick=()=>{playing=false; last=0; document.getElementById('play').textContent='播放'; syncTo(D.duration_ms)};
document.getElementById('spd').onchange=e=>spd=Number(e.target.value);
document.querySelectorAll('button[data-f]').forEach(btn=>btn.onclick=()=>{
  filter=btn.dataset.f;
  document.querySelectorAll('button[data-f]').forEach(b=>b.classList.toggle('on', b===btn));
  rebuildLog();
});
cv.onclick=ev=>{
  const r=cv.getBoundingClientRect();
  playing=false; last=0; document.getElementById('play').textContent='播放';
  syncTo((ev.clientX-r.left)/r.width*D.duration_ms);
};
window.addEventListener('keydown',e=>{ if(e.code==='Space'){ e.preventDefault(); document.getElementById('play').click(); }});
window.addEventListener('resize',resize);
reset(); requestAnimationFrame(tick);
</script></body></html>
"""


def write_demo_html(events, grid, stats, out_path):
    dur = grid[-1]["t"] if grid else 0
    payload = {
        "duration_ms": dur,
        "stats": {
            "gt_single": stats["gt_single"],
            "predictions": stats["predictions"],
            "matched": stats["matched"],
            "joint_ok": stats["joint_ok"],
            "zone_wrong": stats["zone_wrong"],
            "misses": stats["misses"],
            "extras": stats["extras"],
            "hit_recall": stats["hit_recall"],
            "hit_precision": stats["hit_precision"],
            "zone_acc_given_hit": stats["zone_acc_given_hit"],
            "product_recall": stats["product_recall"],
        },
        "events": events,
        "wave": grid,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(HEAD)
        f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        f.write(TAIL)


def main():
    print("Raw CSV 流過現行 HitDetector + 鼓位模型")
    gt = load_gt()
    raw = load_raw_rows()
    samples = rows_to_samples(raw)
    clf = load_zone_model(MODEL_PATH)
    dets = replay_hits(raw)
    hits = classify_detections(dets, samples, clf)
    report = evaluate(gt, hits)
    write_eval_json(report, EVAL_JSON_PATH)
    events = build_events(hits, report)
    write_demo_html(events, energy_grid(samples), report, DEMO_HTML)
    old = os.path.join(CLEANED_DIR, "HoloGrip_Song2_RawCSV打擊與鼓位Demo_0821_ZH_TW.html")
    with open(old, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><meta charset='utf-8'>"
            "<meta http-equiv='refresh' content='0;url=HoloGrip_Song2_產品Demo_即時正確率_0821_ZH_TW.html'>"
            "<p><a href='HoloGrip_Song2_產品Demo_即時正確率_0821_ZH_TW.html'>改開產品 Demo（即時正確率）</a></p>"
        )
    print(
        f"   手+鼓 {report['joint_ok']}/{report['gt_single']} "
        f"({report['product_recall']*100:.1f}%)  "
        f"召回 {report['hit_recall']*100:.1f}%  "
        f"精確率 {report['hit_precision']*100:.1f}%  "
        f"鼓位 {report['zone_acc_given_hit']*100:.1f}%"
    )
    print(DEMO_HTML)
    webbrowser.open(DEMO_HTML)


if __name__ == "__main__":
    main()