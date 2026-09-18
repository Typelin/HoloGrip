"""
Generate HoloGrip 3D Drum Scene + 2D Waveform + Video Sync (v3 Full Rewrite)
=============================================================================
Fixes:
1. Drum L/R orientation corrected (Hi-Hat/Crash LEFT, Ride/Floor Tom RIGHT)
2. Continuous timeline playback (NOT event-hopping)
3. Zero JS errors with all null-checks
"""
import os, csv, json, math

CLEANED_DIR = r'C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Derived\Song_Collection_COM\S20260812_P01_song02_T1127_cleaned'
GT_CSV = os.path.join(CLEANED_DIR, 'Song2_ground_truth_labels_final_0821.csv')
RAW_CSV = r'C:\Users\Typelin_Station\Desktop\HoloGrip\Data\Raw\Song_Collection_COM\S20260812_P01_song02_raw_100hz_20260812_112101.csv'
OUT_HTML = os.path.join(CLEANED_DIR, 'HoloGrip_Song2_3D鼓組空間與打擊重播_0821_ZH_TW.html')

# CORRECTED X signs: camera at (0,y,-z) looking +Z => screen LEFT = +X
DRUMS = {
    "小鼓":     {"x": 0.22, "y":0.74, "z":0.36, "r":0.16, "color":"#00f0ff", "type":"drum", "short":"小鼓 Snare"},
    "高音 Tom": {"x": 0.16, "y":0.98, "z":0.62, "r":0.13, "color":"#38bdf8", "type":"drum", "short":"高音 Tom"},
    "中音 Tom": {"x":-0.16, "y":0.98, "z":0.62, "r":0.13, "color":"#818cf8", "type":"drum", "short":"中音 Tom"},
    "落地 Tom": {"x":-0.48, "y":0.72, "z":0.36, "r":0.18, "color":"#a855f7", "type":"drum", "short":"落地 Tom"},
    "Hi-Hat":   {"x": 0.56, "y":0.92, "z":0.42, "r":0.18, "color":"#eab308", "type":"cymbal","short":"Hi-Hat"},
    "Crash":    {"x": 0.54, "y":1.26, "z":0.68, "r":0.21, "color":"#f97316", "type":"cymbal","short":"Crash 鈸"},
    "Ride":     {"x":-0.56, "y":1.20, "z":0.58, "r":0.23, "color":"#10b981", "type":"cymbal","short":"Ride 鈸"},
}

def main():
    print("Loading ground truth...")
    with open(GT_CSV, 'r', encoding='utf-8') as f:
        gt = list(csv.DictReader(f))

    print("Loading raw 100Hz...")
    samp = {'L':[], 'R':[]}
    with open(RAW_CSV, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            h = r['hand']
            if h in samp:
                samp[h].append({'t':float(r['song_time_ms']),'ax':float(r['ax_g']),'ay':float(r['ay_g']),
                    'az':float(r['az_g']),'mag':float(r['accel_magnitude_g']),
                    'yaw':float(r['cal_yaw_deg']),'pitch':float(r['cal_pitch_deg']),'roll':float(r['roll_deg'])})
    for h in samp: samp[h].sort(key=lambda x:x['t'])

    # Energy grid (10ms bins)
    dur = max(samp['L'][-1]['t'], samp['R'][-1]['t'])
    nbins = int(math.ceil(dur/10))+1
    egrid = [{'t':i*10,'l':0.0,'r':0.0} for i in range(nbins)]
    for h in ['L','R']:
        for s in samp[h]:
            bi = int(round(s['t']/10))
            if 0 <= bi < nbins:
                v = abs(s['mag']-1.0)
                if h=='L': egrid[bi]['l']=max(egrid[bi]['l'],v)
                else: egrid[bi]['r']=max(egrid[bi]['r'],v)

    events = []
    for ev in gt:
        eid=int(ev['event_id']); dn=ev['drum']; ct=float(ev['csv_center_ms'])
        fh=ev['final_hand']; vel=int(ev['velocity'])
        base=DRUMS.get(dn,{"x":0,"y":0.8,"z":0.5,"r":0.15,"color":"#fff","type":"drum","short":dn})

        def closest(hk):
            c=[s for s in samp[hk] if abs(s['t']-ct)<=40]
            if c: c.sort(key=lambda s:s['mag'],reverse=True); return c[0]
            return min(samp[hk],key=lambda s:abs(s['t']-ct))

        li,ri = closest('L'),closest('R')
        ai = li if fh=='L' else (ri if fh=='R' else {k:(li[k]+ri[k])/2 for k in li})
        if fh=='DUAL': ai['mag']=max(li['mag'],ri['mag'])

        yr,pr = math.radians(ai['yaw']),math.radians(ai['pitch'])
        px = base['x'] + math.sin(yr)*0.05 + ai['ax']*0.008
        pz = base['z'] + math.cos(yr)*0.05 + ai['az']*0.008
        py = base['y'] + 0.02 + abs(math.sin(pr))*0.025 + abs(ai['mag']-1.0)*0.004

        events.append({'id':eid,'midi_ms':float(ev['midi_time_ms']),'csv_ms':ct,
            'video_ms':float(ev['midi_time_ms'])+4960.0,'drum':dn,'short':base['short'],
            'note':int(ev['midi_note']),'vel':vel,'hand':fh,
            'pos':[round(px,4),round(py,4),round(pz,4)],
            'base':[base['x'],base['y'],base['z']],'color':base['color'],
            'imu':{'mag':round(ai['mag'],3),'ax':round(ai['ax'],3),'ay':round(ai['ay'],3),
                'az':round(ai['az'],3),'yaw':round(ai['yaw'],1),'pitch':round(ai['pitch'],1),
                'roll':round(ai['roll'],1)},
            'l_mag':round(li['mag'],3),'r_mag':round(ri['mag'],3)})

    print(f"Processed {len(events)} events. Writing HTML...")

    events_json = json.dumps(events, ensure_ascii=False)
    drums_json = json.dumps(DRUMS, ensure_ascii=False)
    egrid_json = json.dumps(egrid, ensure_ascii=False)

    html = build_html(events_json, drums_json, egrid_json)
    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Done: {OUT_HTML}")


def build_html(events_json, drums_json, egrid_json):
    return f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>HoloGrip 3D+2D+Video</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;user-select:none}}
body{{background:#070a12;color:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;overflow:hidden;width:100vw;height:100vh;display:flex;flex-direction:column}}
.hdr{{height:46px;padding:0 16px;background:#0d1322;border-bottom:1px solid rgba(255,255,255,.1);display:flex;justify-content:space-between;align-items:center;z-index:20;flex-shrink:0}}
.hdr .t{{font-size:14px;font-weight:700;color:#fff;display:flex;align-items:center;gap:8px}}
.hdr .badge{{background:linear-gradient(135deg,#2563eb,#38bdf8);color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px}}
.hdr .sub{{font-size:10px;color:#94a3b8}}
.vb{{display:flex;gap:4px}}
.vb button{{background:#1e293b;border:1px solid rgba(255,255,255,.1);color:#94a3b8;padding:4px 9px;font-size:10px;font-weight:600;border-radius:5px;cursor:pointer}}
.vb button:hover,.vb button.on{{background:#2563eb;color:#fff;border-color:#3b82f6}}
.ws{{flex:1;position:relative;overflow:hidden}}
#cv{{width:100%;height:100%;display:block}}
.hud{{position:absolute;top:12px;left:12px;width:260px;background:rgba(13,19,34,.92);border:1px solid rgba(255,255,255,.14);border-radius:8px;padding:10px;backdrop-filter:blur(12px);z-index:15;box-shadow:0 8px 24px rgba(0,0,0,.6)}}
.hud .ht{{font-size:10px;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px;display:flex;justify-content:space-between}}
.hi{{padding:6px;border-radius:5px;font-weight:700;font-size:11px;margin-bottom:6px;text-align:center}}
.hi.L{{background:rgba(37,99,235,.25);color:#60a5fa;border:1px solid #3b82f6}}
.hi.R{{background:rgba(225,29,72,.25);color:#fb7185;border:1px solid #f43f5e}}
.hi.DUAL{{background:rgba(168,85,247,.25);color:#c084fc;border:1px solid #a855f7}}
.sg{{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:6px}}
.sc{{background:rgba(26,36,56,.6);padding:4px 6px;border-radius:5px;border:1px solid rgba(255,255,255,.05)}}
.sc .sl{{font-size:8px;color:#64748b;font-weight:600}}
.sc .sv{{font-size:11px;font-weight:700;color:#f8fafc}}
.ib{{background:rgba(10,15,28,.85);border:1px solid rgba(56,189,248,.25);border-radius:5px;padding:6px;margin-bottom:5px}}
.ir{{display:flex;justify-content:space-between;font-size:10px;margin-bottom:1px;font-variant-numeric:tabular-nums}}
.ir:last-child{{margin-bottom:0}}
.ir .tg{{color:#94a3b8}}
.ir .nm{{font-weight:700;color:#38bdf8}}
.pip{{position:absolute;top:12px;right:12px;width:300px;min-width:200px;min-height:150px;background:rgba(13,19,34,.95);border:1px solid rgba(255,255,255,.2);border-radius:8px;overflow:hidden;z-index:15;box-shadow:0 12px 36px rgba(0,0,0,.7);resize:both}}
.pip .ph{{padding:4px 8px;background:#1e293b;display:flex;justify-content:space-between;align-items:center;font-size:10px;font-weight:700;color:#cbd5e1;cursor:move}}
.pip video{{width:100%;height:calc(100% - 40px);min-height:110px;object-fit:contain;background:#000;display:block}}
.pip .pb{{padding:2px 8px;font-size:9px;color:#94a3b8;display:flex;justify-content:space-between;background:#0f172a}}
.hc{{position:absolute;background:rgba(15,23,42,.96);border:1px solid #38bdf8;border-radius:7px;padding:8px 10px;font-size:10px;pointer-events:none;display:none;z-index:30;box-shadow:0 6px 20px rgba(0,0,0,.7);max-width:240px}}
.dl{{position:absolute;background:rgba(15,23,42,.85);border:1px solid rgba(255,255,255,.2);color:#fff;font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;pointer-events:none;white-space:nowrap;transform:translate(-50%,-50%)}}
.dl.on{{background:#2563eb;border-color:#60a5fa;box-shadow:0 0 8px #3b82f6}}
.bot{{background:#0d1322;border-top:1px solid rgba(255,255,255,.12);padding:6px 14px 10px;z-index:20;display:flex;flex-direction:column;gap:5px;flex-shrink:0}}
.ww{{width:100%;height:75px;background:#080c14;border:1px solid #1e293b;border-radius:5px;overflow:hidden;cursor:crosshair}}
.ww svg{{width:100%;height:100%;display:block}}
.cr{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px}}
.tb{{font-size:12px;font-variant-numeric:tabular-nums;font-weight:700;color:#38bdf8;display:flex;align-items:center;gap:8px}}
.bg{{display:flex;align-items:center;gap:4px}}
.cb{{background:#1e293b;border:1px solid rgba(255,255,255,.1);color:#f8fafc;padding:4px 9px;border-radius:5px;font-size:11px;font-weight:600;cursor:pointer}}
.cb:hover{{background:#334155;color:#38bdf8}}
.cb.p{{background:#2563eb;border-color:#3b82f6}}
.ss{{background:#1e293b;border:1px solid rgba(255,255,255,.1);color:#f8fafc;padding:3px 5px;border-radius:5px;font-size:10px;cursor:pointer}}
.tg2{{background:rgba(30,41,59,.6);border:1px solid rgba(255,255,255,.1);color:#94a3b8;padding:3px 7px;border-radius:5px;font-size:10px;cursor:pointer}}
.tg2.on{{background:rgba(16,185,129,.2);color:#34d399;border-color:#10b981}}
</style>
</head>
<body>
<div class="hdr">
 <div class="t"><span class="badge">HoloGrip 3D+2D</span> 3D 鼓組 · 2D IMU 波形 · 影片全同步 <span class="sub">MIDI→CSV +6,027ms ｜ MIDI→影片 +4,960ms ｜ 窗口±80ms</span></div>
 <div class="vb">
  <button class="on" id="bv0" onclick="setView('pov',this)">鼓手視角 (POV)</button>
  <button id="bv1" onclick="setView('45',this)">斜側45°</button>
  <button id="bv2" onclick="setView('top',this)">俯視</button>
  <button id="bv3" onclick="setView('front',this)">觀眾視角</button>
 </div>
</div>
<div class="ws" id="ws">
 <canvas id="cv"></canvas>
 <div id="lc"></div>
 <div class="hud" id="hud">
  <div class="ht"><span>當前擊打遙測</span><span id="hi0" style="color:#94a3b8">#1/311</span></div>
  <div class="hi R" id="hi1">右手打擊 (R)</div>
  <div class="sg">
   <div class="sc"><div class="sl">鼓件</div><div class="sv" id="h_dr">Hi-Hat</div></div>
   <div class="sc"><div class="sl">力度</div><div class="sv" id="h_ve">94/127</div></div>
   <div class="sc"><div class="sl">MIDI時間</div><div class="sv" id="h_mt">4.324s</div></div>
   <div class="sc"><div class="sl">影片時間</div><div class="sv" id="h_vt">9.284s</div></div>
  </div>
  <div class="ib">
   <div style="font-size:8px;font-weight:700;color:#cbd5e1;margin-bottom:2px;display:flex;justify-content:space-between"><span>手套IMU</span><span id="h_ht" style="color:#38bdf8">R</span></div>
   <div class="ir"><span class="tg">Mag:</span><span class="nm" id="t_m">1.82g</span></div>
   <div class="ir"><span class="tg">Ax/Ay/Az:</span><span class="nm" id="t_a">0.15/1.72/0.54</span></div>
   <div class="ir"><span class="tg">Yaw:</span><span class="nm" id="t_y">-32.4°</span></div>
   <div class="ir"><span class="tg">Pitch:</span><span class="nm" id="t_p">+14.2°</span></div>
   <div class="ir"><span class="tg">Roll:</span><span class="nm" id="t_r">-8.6°</span></div>
  </div>
  <div style="font-size:8px;color:#64748b;display:flex;justify-content:space-between"><span>雙手動能</span><span id="t_as">R:88% L:12%</span></div>
  <div style="height:3px;background:#1e293b;border-radius:2px;overflow:hidden;display:flex;margin:3px 0">
   <div id="bl" style="width:12%;background:#3b82f6"></div>
   <div id="br" style="width:88%;background:#f43f5e"></div>
  </div>
 </div>
 <div class="pip" id="pip">
  <div class="ph" id="piph"><span>📹 影片同步</span><div style="display:flex;gap:5px"><label style="color:#38bdf8;cursor:pointer;font-size:9px">[選檔]<input type="file" id="vfp" accept="video/*" style="display:none"></label><span style="cursor:pointer" onclick="togglePip()">✖</span></div></div>
  <video id="vid" src="./S20260812_P01_song02_T1127_video_h264.mp4" playsinline muted preload="auto"></video>
  <div class="pb"><span>Offset +4,960ms</span><span id="vts">00:00.000</span></div>
 </div>
 <div class="hc" id="hc"></div>
</div>
<div class="bot">
 <div class="ww"><svg id="wsvg" viewBox="0 0 1000 75" preserveAspectRatio="none"></svg></div>
 <div class="cr">
  <div class="tb"><span id="tdisp">00:04.324</span><span style="font-size:10px;color:#94a3b8;font-weight:normal">/01:10</span><span id="htag" style="font-size:10px;color:#e2e8f0;background:#1e293b;padding:2px 5px;border-radius:3px">#1 Hi-Hat (R)</span></div>
  <div class="bg">
   <button class="cb" onclick="stepHit(-1)">◀上一擊</button>
   <button class="cb p" id="pbtn" onclick="togglePlay()">▶ 播放</button>
   <button class="cb" onclick="stepHit(1)">下一擊▶</button>
   <button class="cb" onclick="loopWindow()">🔁 循環±80ms</button>
   <button class="cb" onclick="stopLoop()">⏹ 停止循環</button>
   <button class="cb" onclick="restart()">↺ 重頭</button>
  </div>
  <div class="bg">
   <span style="font-size:10px;color:#94a3b8">速度:</span>
   <select class="ss" id="spd" onchange="setSpeed()"><option value="1">1x</option><option value="0.5" selected>0.5x</option><option value="0.25">0.25x</option><option value="0.125">0.125x</option></select>
   <button class="tg2 on" id="tpc" onclick="togPC()">●點雲</button>
   <button class="tg2 on" id="tst" onclick="togSt()">🥢鼓棒</button>
   <button class="tg2 on" id="tvd" onclick="togglePip()">📹影片</button>
  </div>
 </div>
</div>
<script>
"use strict";
const EV={events_json};
const DS={drums_json};
const EG={egrid_json};

let scene,cam,ren,ctrl,ray,mouse;
let dMesh={{}},pcGroup,lPiv,rPiv,lStk,rStk;
let labEls={{}};

// === MASTER TIMELINE ===
let masterMs=EV[0].midi_ms; // current time in MIDI-ms space
let playing=false, speed=0.5, lastT=0;
let lastTrig=-1; // index of last triggered event
let loopWin=null; // null or {{s,e}} for window loop
const maxMs=EV[EV.length-1].midi_ms+2000;
const vid=document.getElementById('vid');

function init(){{
 const c=document.getElementById('cv');
 scene=new THREE.Scene();
 scene.background=new THREE.Color(0x070a12);
 scene.fog=new THREE.FogExp2(0x070a12,0.22);
 cam=new THREE.PerspectiveCamera(46,c.clientWidth/c.clientHeight,0.05,50);
 ren=new THREE.WebGLRenderer({{canvas:c,antialias:true}});
 ren.setSize(c.clientWidth,c.clientHeight);
 ren.setPixelRatio(Math.min(devicePixelRatio,2));
 ren.shadowMap.enabled=true;
 ctrl=new THREE.OrbitControls(cam,ren.domElement);
 ctrl.enableDamping=true;ctrl.dampingFactor=0.05;
 scene.add(new THREE.AmbientLight(0xffffff,0.7));
 const dl=new THREE.DirectionalLight(0xffffff,0.9);dl.position.set(2,4.5,3);dl.castShadow=true;scene.add(dl);
 scene.add(new THREE.PointLight(0x38bdf8,0.5,6).translateX(-2).translateY(2));
 const g=new THREE.GridHelper(10,30,0x1e293b,0x0c1322);scene.add(g);
 const rug=new THREE.Mesh(new THREE.CircleGeometry(1.6,64),new THREE.MeshStandardMaterial({{color:0x090e1c,roughness:0.9}}));
 rug.rotation.x=-Math.PI/2;scene.add(rug);

 buildKit();buildSticks();buildCloud();buildLabels();
 setView('pov',document.getElementById('bv0'));

 window.addEventListener('resize',onResize);
 document.getElementById('ws').addEventListener('mousemove',onMM);
 document.getElementById('ws').addEventListener('click',onMC);
 document.getElementById('vfp').addEventListener('change',e=>{{const f=e.target.files[0];if(f){{vid.src=URL.createObjectURL(f);vid.load()}}}});
 setupDrag();

 hudUpdate(EV[0],0);
 drawWave(masterMs);
 requestAnimationFrame(tick);
}}

// === 3D BUILD ===
function buildKit(){{
 const grp=new THREE.Group();scene.add(grp);
 // stool
 const stool=new THREE.Mesh(new THREE.CylinderGeometry(.18,.18,.08,32),new THREE.MeshStandardMaterial({{color:0x1e293b}}));
 stool.position.set(0,.5,-.2);grp.add(stool);
 // kick
 const kick=new THREE.Mesh(new THREE.CylinderGeometry(.32,.32,.38,32),new THREE.MeshStandardMaterial({{color:0x1e293b,metalness:.3}}));
 kick.rotation.x=Math.PI/2;kick.position.set(0,.35,.72);grp.add(kick);

 for(const[n,s]of Object.entries(DS)){{
  const g2=new THREE.Group();g2.position.set(s.x,s.y,s.z);
  if(s.type==='drum'){{
   g2.add(new THREE.Mesh(new THREE.CylinderGeometry(s.r,s.r,.08,32),new THREE.MeshStandardMaterial({{color:0x1e293b,metalness:.5,roughness:.4}})));
   const hd=new THREE.Mesh(new THREE.CylinderGeometry(s.r*.92,s.r*.92,.082,32),new THREE.MeshStandardMaterial({{color:new THREE.Color(s.color),roughness:.35,emissive:new THREE.Color(s.color),emissiveIntensity:.2}}));
   g2.add(hd);
   const st=new THREE.Mesh(new THREE.CylinderGeometry(.012,.012,s.y,16),new THREE.MeshStandardMaterial({{color:0x64748b,metalness:.8}}));
   st.position.y=-s.y/2;g2.add(st);
   dMesh[n]={{hd,defEm:.2}};
  }}else{{
   const cy=new THREE.Mesh(new THREE.ConeGeometry(s.r,.038,32,1,true),new THREE.MeshStandardMaterial({{color:new THREE.Color(s.color),metalness:.88,roughness:.2,side:THREE.DoubleSide,emissive:new THREE.Color(s.color),emissiveIntensity:.25}}));
   cy.rotation.x=Math.PI;g2.add(cy);
   const st=new THREE.Mesh(new THREE.CylinderGeometry(.012,.012,s.y,16),new THREE.MeshStandardMaterial({{color:0x64748b,metalness:.8}}));
   st.position.y=-s.y/2;g2.add(st);
   dMesh[n]={{hd:cy,defEm:.25}};
  }}
  grp.add(g2);
 }}
}}

function buildSticks(){{
 const sg=new THREE.CylinderGeometry(.006,.014,.36,16);sg.translate(0,.18,0);
 lPiv=new THREE.Group();lPiv.position.set(.22,.98,.15);scene.add(lPiv);
 lStk=new THREE.Mesh(sg,new THREE.MeshStandardMaterial({{color:0x3b82f6,emissive:0x2563eb,emissiveIntensity:.45}}));
 lStk.rotation.x=Math.PI/3.5;lStk.rotation.z=-Math.PI/12;lPiv.add(lStk);
 rPiv=new THREE.Group();rPiv.position.set(-.22,.98,.15);scene.add(rPiv);
 rStk=new THREE.Mesh(sg.clone(),new THREE.MeshStandardMaterial({{color:0xf43f5e,emissive:0xe11d48,emissiveIntensity:.45}}));
 rStk.rotation.x=Math.PI/3.5;rStk.rotation.z=Math.PI/12;rPiv.add(rStk);
}}

function buildCloud(){{
 pcGroup=new THREE.Group();scene.add(pcGroup);
 const sg=new THREE.SphereGeometry(.013,14,14);
 EV.forEach((e,i)=>{{
  const c=e.hand==='L'?0x3b82f6:e.hand==='R'?0xf43f5e:0xa855f7;
  const m=new THREE.Mesh(sg,new THREE.MeshStandardMaterial({{color:c,emissive:c,emissiveIntensity:.45,roughness:.25}}));
  m.position.set(e.pos[0],e.pos[1],e.pos[2]);
  m.userData={{ev:e,idx:i}};
  pcGroup.add(m);
 }});
}}

function buildLabels(){{
 const lc=document.getElementById('lc');lc.innerHTML='';
 for(const[n,s]of Object.entries(DS)){{
  const el=document.createElement('div');el.className='dl';el.innerText=s.short;lc.appendChild(el);
  labEls[n]={{el,pos:new THREE.Vector3(s.x,s.y+.12,s.z)}};
 }}
}}

function updLabels(){{
 if(!cam||!ren)return;
 const cv=ren.domElement;
 for(const[n,d]of Object.entries(labEls)){{
  const wp=d.pos.clone().project(cam);
  if(wp.z>1){{d.el.style.display='none';continue}}
  d.el.style.display='block';
  d.el.style.left=((wp.x*.5+.5)*cv.clientWidth)+'px';
  d.el.style.top=((-(wp.y*.5)+.5)*cv.clientHeight)+'px';
 }}
}}

// === TRIGGER HIT ===
function trigHit(ev,idx){{
 const d=dMesh[ev.drum];
 if(d){{d.hd.material.emissiveIntensity=1.5;setTimeout(()=>{{d.hd.material.emissiveIntensity=d.defEm}},120)}}
 Object.values(labEls).forEach(l=>l.el.classList.remove('on'));
 if(labEls[ev.drum])labEls[ev.drum].el.classList.add('on');
 const tp=ev.pos;
 if(ev.hand==='L'||ev.hand==='DUAL'){{
  lPiv.position.set(tp[0]*.7,tp[1]+.18,tp[2]-.22);
  lPiv.lookAt(tp[0],tp[1]+.02,tp[2]);
  lStk.rotation.x=Math.PI/2.2;setTimeout(()=>{{lStk.rotation.x=Math.PI/3.4}},60);
 }}
 if(ev.hand==='R'||ev.hand==='DUAL'){{
  rPiv.position.set(tp[0]*.7,tp[1]+.18,tp[2]-.22);
  rPiv.lookAt(tp[0],tp[1]+.02,tp[2]);
  rStk.rotation.x=Math.PI/2.2;setTimeout(()=>{{rStk.rotation.x=Math.PI/3.4}},60);
 }}
 hudUpdate(ev,idx);
}}

// === MAIN ANIMATION LOOP (continuous timeline) ===
function tick(ts){{
 requestAnimationFrame(tick);

 // advance master time
 if(playing){{
  if(!lastT)lastT=ts;
  const dt=ts-lastT;
  masterMs+=dt*speed;
  lastT=ts;

  // window loop
  if(loopWin&&masterMs>loopWin.e)masterMs=loopWin.s;
  // end of song
  if(masterMs>maxMs){{masterMs=maxMs;pause()}}
 }}else{{
  lastT=0;
 }}

 // sync video
 syncVid();
 // trigger events that we passed
 trigPassed();
 // update waveform
 drawWave(masterMs);
 // time display
 document.getElementById('tdisp').innerText=fmtT(masterMs);

 if(ctrl)ctrl.update();
 updLabels();
 if(ren&&scene&&cam)ren.render(scene,cam);
}}

function syncVid(){{
 const vt=(masterMs+4960)/1000;
 document.getElementById('vts').innerText=fmtT(vt*1000);
 if(vid.readyState<2)return;
 if(playing&&vid.paused)vid.play().catch(()=>{{}});
 if(!playing&&!vid.paused)vid.pause();
 vid.playbackRate=speed;
 if(Math.abs(vid.currentTime-vt)>.2)vid.currentTime=vt;
}}

function trigPassed(){{
 // trigger any events whose midi_ms <= masterMs that haven't been triggered
 for(let i=lastTrig+1;i<EV.length;i++){{
  if(EV[i].midi_ms<=masterMs){{
   trigHit(EV[i],i);lastTrig=i;
  }}else break;
 }}
 // update tag to nearest
 const ne=nearestEv();
 if(ne)document.getElementById('htag').innerText=`#${{ne.idx+1}} ${{ne.ev.drum}} (${{ne.ev.hand==='L'?'左':'右'}})`;
}}

function nearestEv(){{
 let best=null,bd=Infinity;
 EV.forEach((e,i)=>{{const d=Math.abs(e.midi_ms-masterMs);if(d<bd){{bd=d;best={{ev:e,idx:i}}}}}});
 return best;
}}

// === 2D WAVEFORM ===
function drawWave(tMs){{
 const svg=document.getElementById('wsvg');if(!svg)return;
 const csvT=tMs+6027; // convert MIDI-ms to CSV-ms
 const look=240;
 const s0=csvT-look,s1=csvT+look;
 const sl=EG.filter(p=>p.t>=s0&&p.t<=s1);
 const mx=Math.max(1.5,...sl.map(p=>Math.max(p.l,p.r)));
 const gx=t=>50+((t-s0)/(look*2))*900;
 const gy=v=>65-(v/mx)*52;
 const lp=sl.map(p=>`${{gx(p.t).toFixed(0)}},${{gy(p.l).toFixed(0)}}`).join(' ');
 const rp=sl.map(p=>`${{gx(p.t).toFixed(0)}},${{gy(p.r).toFixed(0)}}`).join(' ');
 const wl=gx(csvT-80),ww=gx(csvT+80)-wl;
 // nearby MIDI markers
 const nb=EV.filter(e=>Math.abs(e.csv_ms-csvT)<=look);
 const mk=nb.map(e=>{{
  const px=gx(e.csv_ms);
  return `<rect x="${{(px-2).toFixed(0)}}" y="4" width="4" height="8" rx="1" fill="${{Math.abs(e.csv_ms-csvT)<5?'#38bdf8':'#64748b'}}"><title>#${{e.id}} ${{e.drum}}</title></rect>`;
 }}).join('');
 // playhead (current time position) - always at center
 svg.innerHTML=`
  <rect x="50" y="6" width="900" height="60" fill="#0b1120"/>
  <rect x="${{wl.toFixed(0)}}" y="6" width="${{ww.toFixed(0)}}" height="60" fill="rgba(16,185,129,.12)" stroke="rgba(16,185,129,.35)" stroke-dasharray="2 2"/>
  <line x1="50" y1="65" x2="950" y2="65" stroke="#1e293b"/>
  <polyline points="${{lp}}" fill="none" stroke="#3b82f6" stroke-width="1.8"/>
  <polyline points="${{rp}}" fill="none" stroke="#f43f5e" stroke-width="1.8"/>
  <line x1="500" y1="6" x2="500" y2="65" stroke="#fff" stroke-width="1.5"/>
  ${{mk}}
  <text x="54" y="16" fill="#3b82f6" font-size="9" font-weight="700">L</text>
  <text x="68" y="16" fill="#f43f5e" font-size="9" font-weight="700">R</text>
  <text x="86" y="16" fill="#10b981" font-size="9">±80ms</text>
 `;
}}

// === HUD ===
function hudUpdate(ev,idx){{
 document.getElementById('hi0').innerText=`#${{idx+1}}/${{EV.length}}`;
 const hel=document.getElementById('hi1');
 hel.className='hi '+ev.hand;
 hel.innerText=ev.hand==='L'?'左手打擊 (L)':ev.hand==='R'?'右手打擊 (R)':'雙手齊擊 (DUAL)';
 document.getElementById('h_dr').innerText=ev.short;
 document.getElementById('h_ve').innerText=ev.vel+'/127';
 document.getElementById('h_mt').innerText=(ev.midi_ms/1000).toFixed(3)+'s';
 document.getElementById('h_vt').innerText=(ev.video_ms/1000).toFixed(3)+'s';
 document.getElementById('h_ht').innerText=ev.hand;
 const m=ev.imu;
 document.getElementById('t_m').innerText=m.mag+'g';
 document.getElementById('t_a').innerText=m.ax+'/'+m.ay+'/'+m.az;
 document.getElementById('t_y').innerText=(m.yaw>0?'+':'')+m.yaw+'°';
 document.getElementById('t_p').innerText=(m.pitch>0?'+':'')+m.pitch+'°';
 document.getElementById('t_r').innerText=(m.roll>0?'+':'')+m.roll+'°';
 const lm=ev.l_mag,rm=ev.r_mag,tot=lm+rm+.001;
 const lp=Math.round(lm/tot*100),rp=100-lp;
 document.getElementById('t_as').innerText=`R:${{rp}}% L:${{lp}}%`;
 document.getElementById('bl').style.width=lp+'%';
 document.getElementById('br').style.width=rp+'%';
}}

// === CONTROLS ===
function togglePlay(){{if(playing)pause();else play()}}
function play(){{playing=true;lastT=0;document.getElementById('pbtn').innerText='⏸ 暫停'}}
function pause(){{playing=false;vid.pause();document.getElementById('pbtn').innerText='▶ 播放'}}
function restart(){{masterMs=EV[0].midi_ms;lastTrig=-1;play()}}
function setSpeed(){{speed=parseFloat(document.getElementById('spd').value)}}

function stepHit(d){{
 pause();loopWin=null;
 const ne=nearestEv();if(!ne)return;
 let ni=ne.idx+d;
 if(ni<0)ni=0;if(ni>=EV.length)ni=EV.length-1;
 masterMs=EV[ni].midi_ms;lastTrig=ni-1;
 trigHit(EV[ni],ni);drawWave(masterMs);
}}

function loopWindow(){{
 const ne=nearestEv();if(!ne)return;
 loopWin={{s:ne.ev.midi_ms-80,e:ne.ev.midi_ms+80}};
 masterMs=loopWin.s;lastTrig=ne.idx-1;
 play();
}}
function stopLoop(){{loopWin=null;pause()}}

function togglePip(){{
 const p=document.getElementById('pip');
 const on=p.style.display!=='none';
 p.style.display=on?'none':'block';
 document.getElementById('tvd').classList.toggle('on',!on);
}}

function togPC(){{
 if(pcGroup)pcGroup.visible=!pcGroup.visible;
 document.getElementById('tpc').classList.toggle('on',pcGroup&&pcGroup.visible);
}}
function togSt(){{
 if(lPiv)lPiv.visible=!lPiv.visible;
 if(rPiv)rPiv.visible=!rPiv.visible;
 document.getElementById('tst').classList.toggle('on',lPiv&&lPiv.visible);
}}

// === CAMERA ===
function setView(v,btn){{
 document.querySelectorAll('.vb button').forEach(b=>b.classList.remove('on'));
 if(btn)btn.classList.add('on');
 if(!cam||!ctrl)return;
 if(v==='pov'){{cam.position.set(0,1.25,-.35);ctrl.target.set(0,.88,.65)}}
 else if(v==='45'){{cam.position.set(-.8,1.8,-1.5);ctrl.target.set(0,.85,.45)}}
 else if(v==='top'){{cam.position.set(0,3.2,.45);ctrl.target.set(0,0,.45)}}
 else if(v==='front'){{cam.position.set(0,1.15,2.7);ctrl.target.set(0,.9,.4)}}
}}

// === HOVER ===
ray=new THREE.Raycaster();mouse=new THREE.Vector2();
function onMM(e){{
 if(!ren||!cam||!pcGroup||!pcGroup.visible)return;
 const cv=ren.domElement,rc=cv.getBoundingClientRect();
 mouse.x=((e.clientX-rc.left)/cv.clientWidth)*2-1;
 mouse.y=-((e.clientY-rc.top)/cv.clientHeight)*2+1;
 ray.setFromCamera(mouse,cam);
 const its=ray.intersectObjects(pcGroup.children);
 const hc=document.getElementById('hc');
 if(its.length>0){{
  const h=its[0].object.userData.ev,i=its[0].object.userData.idx;
  hc.style.display='block';hc.style.left=(e.clientX+12)+'px';hc.style.top=(e.clientY+12)+'px';
  hc.innerHTML=`<div style="font-weight:700;color:#38bdf8;margin-bottom:3px">#${{i+1}} ${{h.drum}} <span style="color:${{h.hand==='L'?'#60a5fa':'#fb7185'}}">${{h.hand}}</span></div><div style="color:#cbd5e1;line-height:1.4">⏱${{(h.midi_ms/1000).toFixed(3)}}s vel:${{h.vel}}<br>⚡${{h.imu.mag}}g Ax:${{h.imu.ax}} Ay:${{h.imu.ay}}<br>🧭Yaw:${{h.imu.yaw}}° Pitch:${{h.imu.pitch}}°</div>`;
 }}else hc.style.display='none';
}}
function onMC(e){{
 if(!pcGroup||!pcGroup.visible)return;
 ray.setFromCamera(mouse,cam);
 const its=ray.intersectObjects(pcGroup.children);
 if(its.length>0){{
  const i=its[0].object.userData.idx;
  pause();masterMs=EV[i].midi_ms;lastTrig=i-1;
  trigHit(EV[i],i);drawWave(masterMs);
 }}
}}

// === DRAG PIP ===
function setupDrag(){{
 const p=document.getElementById('pip'),h=document.getElementById('piph');
 let dr=false,sx,sy,il,it;
 h.addEventListener('mousedown',e=>{{if(e.target.tagName==='INPUT'||e.target.tagName==='LABEL')return;dr=true;sx=e.clientX;sy=e.clientY;il=p.offsetLeft;it=p.offsetTop;document.body.style.cursor='move'}});
 window.addEventListener('mousemove',e=>{{if(!dr)return;p.style.left=(il+e.clientX-sx)+'px';p.style.top=(it+e.clientY-sy)+'px';p.style.right='auto'}});
 window.addEventListener('mouseup',()=>{{dr=false;document.body.style.cursor='default'}});
}}

function fmtT(ms){{const s=Math.floor(ms/1000),m=Math.floor(s/60),ss=s%60,ml=Math.floor(ms%1000);return String(m).padStart(2,'0')+':'+String(ss).padStart(2,'0')+'.'+String(ml).padStart(3,'0')}}
function onResize(){{if(!ren||!cam)return;const c=ren.domElement;cam.aspect=c.clientWidth/c.clientHeight;cam.updateProjectionMatrix();ren.setSize(c.clientWidth,c.clientHeight)}}

window.onload=init;
</script>
</body>
</html>'''


if __name__=='__main__':
    main()
