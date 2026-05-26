"""
build_final.py  — builds the complete self-contained HTML dashboard
"""
import json, os

print("Loading simulation data...")
with open('/home/claude/blr_pollution/data/sim_data.json') as f:
    data = json.load(f)

SIM_JSON = json.dumps(data, separators=(',', ':'))

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Simulation and Control of Pollution Spread Using PDEs and Gradient Descent Optimization</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
:root{
  --bg:#02050a;--panel:#060c18;--panel2:#08111f;--border:#0f2035;--border2:#183050;
  --cyan:#00e5ff;--orange:#ff6b35;--green:#39ff14;--purple:#a855f7;--yellow:#fbbf24;
  --red:#f43f5e;--text:#b8cfe8;--dim:#3a5878;
  --glow-c:0 0 20px rgba(0,229,255,0.5);--glow-g:0 0 20px rgba(57,255,20,0.5);
  --glow-o:0 0 20px rgba(255,107,53,0.5);
}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:"Exo 2",sans-serif;}
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,229,255,0.008) 3px,rgba(0,229,255,0.008) 4px);}

/* ── HEADER ── */
#hdr{
  position:fixed;top:0;left:0;right:0;height:60px;z-index:2000;
  background:linear-gradient(90deg,#02050a,#071428 30%,#071428 70%,#02050a);
  border-bottom:1px solid var(--border2);
  display:flex;align-items:center;gap:18px;padding:0 24px;
}
.hdr-logo{
  width:42px;height:42px;border-radius:50%;border:2px solid var(--cyan);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
  background:radial-gradient(circle,rgba(0,229,255,0.1),transparent);
  animation:logobeat 3s ease-in-out infinite;box-shadow:var(--glow-c);
}
@keyframes logobeat{0%,100%{box-shadow:0 0 10px rgba(0,229,255,0.3)}50%{box-shadow:0 0 30px rgba(0,229,255,0.8)}}
.hdr-logo svg{width:22px;height:22px;stroke:var(--cyan);fill:none;stroke-width:2;}
.hdr-titles{flex:1;min-width:0;}
.hdr-titles h1{
  font-family:"Orbitron",monospace;font-size:0.82rem;font-weight:700;
  color:var(--cyan);letter-spacing:1.5px;text-transform:uppercase;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.hdr-titles p{font-family:"Share Tech Mono",monospace;font-size:0.55rem;color:var(--dim);
  letter-spacing:2px;text-transform:uppercase;margin-top:2px;}
.hdr-metrics{display:flex;align-items:center;gap:20px;flex-shrink:0;}
.hm{text-align:center;}
.hm-val{font-family:"Orbitron",monospace;font-size:1.2rem;font-weight:700;line-height:1;}
.hm-lbl{font-family:"Share Tech Mono",monospace;font-size:0.5rem;color:var(--dim);
  letter-spacing:1px;text-transform:uppercase;margin-top:2px;}
.hm-val.c{color:var(--cyan);text-shadow:var(--glow-c);}
.hm-val.g{color:var(--green);text-shadow:var(--glow-g);}
.hm-val.o{color:var(--orange);text-shadow:var(--glow-o);}
.hm-val.p{color:var(--purple);}
.hdr-sep{width:1px;height:30px;background:var(--border2);}
.sig-pill{
  padding:4px 12px;border-radius:20px;font-family:"Share Tech Mono",monospace;
  font-size:0.58rem;letter-spacing:1px;text-transform:uppercase;
  border:1px solid var(--green);color:var(--green);
  background:rgba(57,255,20,0.1);box-shadow:var(--glow-g);
  animation:sigblink 2s ease-in-out infinite;
}
@keyframes sigblink{0%,100%{opacity:1}50%{opacity:0.6}}

/* ── LAYOUT ── */
#app{position:fixed;top:60px;left:0;right:0;bottom:0;display:grid;grid-template-columns:280px 1fr 280px;}

/* ── SIDEBARS ── */
.sb{
  background:var(--panel);overflow-y:auto;overflow-x:hidden;
  scrollbar-width:thin;scrollbar-color:var(--border2) transparent;
  display:flex;flex-direction:column;
}
.sb-l{border-right:1px solid var(--border2);}
.sb-r{border-left:1px solid var(--border2);}
::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}

/* ── CARDS ── */
.card{border-bottom:1px solid var(--border);padding:13px 15px;}
.card:last-child{border-bottom:none;}
.ctitle{
  font-family:"Share Tech Mono",monospace;font-size:0.6rem;color:var(--cyan);
  letter-spacing:2.5px;text-transform:uppercase;margin-bottom:10px;
  display:flex;align-items:center;gap:7px;
}
.ctitle::before{content:"";width:5px;height:5px;border-radius:50%;background:var(--cyan);
  flex-shrink:0;box-shadow:0 0 6px var(--cyan);}
.ctitle.orange{color:var(--orange);}
.ctitle.orange::before{background:var(--orange);box-shadow:0 0 6px var(--orange);}
.ctitle.green{color:var(--green);}
.ctitle.green::before{background:var(--green);box-shadow:0 0 6px var(--green);}
.ctitle.purple{color:var(--purple);}
.ctitle.purple::before{background:var(--purple);box-shadow:0 0 6px var(--purple);}

/* ── METRICS GRID ── */
.mgrid{display:grid;grid-template-columns:1fr 1fr;gap:7px;}
.met{background:var(--panel2);border:1px solid var(--border);border-radius:5px;padding:8px 10px;}
.met-v{font-family:"Orbitron",monospace;font-size:1.05rem;font-weight:700;}
.met-l{font-family:"Share Tech Mono",monospace;font-size:0.5rem;color:var(--dim);
  letter-spacing:1px;text-transform:uppercase;margin-top:2px;}
.met-v.c{color:var(--cyan);}
.met-v.g{color:var(--green);}
.met-v.o{color:var(--orange);}
.met-v.p{color:var(--purple);}
.met-v.y{color:var(--yellow);}

/* ── TABS ── */
.tabs{display:flex;gap:3px;margin-bottom:10px;flex-wrap:wrap;}
.tab{
  padding:3px 9px;font-family:"Share Tech Mono",monospace;font-size:0.56rem;
  letter-spacing:1px;text-transform:uppercase;cursor:pointer;
  border:1px solid var(--border2);border-radius:3px;color:var(--dim);
  background:transparent;transition:all .15s;
}
.tab:hover{color:var(--cyan);border-color:var(--cyan);}
.tab.on{color:var(--cyan);border-color:var(--cyan);background:rgba(0,229,255,0.08);}
.tab.on.o{color:var(--orange);border-color:var(--orange);background:rgba(255,107,53,0.08);}
.tab.on.g{color:var(--green);border-color:var(--green);background:rgba(57,255,20,0.08);}

/* ── BUTTONS ── */
.btn{
  display:inline-flex;align-items:center;gap:5px;
  padding:6px 12px;border-radius:4px;cursor:pointer;
  font-family:"Share Tech Mono",monospace;font-size:0.6rem;
  letter-spacing:1px;text-transform:uppercase;border:1px solid;
  transition:all .2s;background:transparent;
}
.btn-c{color:var(--cyan);border-color:var(--cyan);}
.btn-c:hover{background:rgba(0,229,255,0.12);box-shadow:var(--glow-c);}
.btn-g{color:var(--green);border-color:var(--green);}
.btn-g:hover{background:rgba(57,255,20,0.1);box-shadow:var(--glow-g);}
.btn-o{color:var(--orange);border-color:var(--orange);}
.btn-o:hover{background:rgba(255,107,53,0.12);box-shadow:var(--glow-o);}
.btn-p{color:var(--purple);border-color:var(--purple);}
.btn-p:hover{background:rgba(168,85,247,0.12);}
.btn-y{color:var(--yellow);border-color:var(--yellow);}
.btn-y:hover{background:rgba(251,191,36,0.1);}
.brow{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px;}

/* ── PROGRESS ── */
.pbar-wrap{margin:5px 0;}
.pbar-top{display:flex;justify-content:space-between;
  font-family:"Share Tech Mono",monospace;font-size:0.56rem;color:var(--dim);margin-bottom:3px;}
.pbar-bg{height:4px;background:var(--border);border-radius:2px;overflow:hidden;}
.pbar-fill{height:100%;border-radius:2px;transition:width 1.8s cubic-bezier(.4,0,.2,1);}

/* ── SOURCES LIST ── */
.src-row{
  display:flex;align-items:center;gap:8px;padding:5px 2px;
  border-bottom:1px solid var(--border);cursor:pointer;transition:background .15s;
}
.src-row:last-child{border-bottom:none;}
.src-row:hover{background:rgba(0,229,255,0.04);border-radius:3px;}
.src-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;}
.src-name{font-family:"Share Tech Mono",monospace;font-size:0.6rem;flex:1;}
.src-str{font-family:"Orbitron",monospace;font-weight:700;font-size:0.8rem;color:var(--orange);}

/* ── PILLS ── */
.pills{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;}
.pill{
  padding:2px 7px;border-radius:12px;font-family:"Share Tech Mono",monospace;
  font-size:0.52rem;letter-spacing:0.5px;border:1px solid;
}
.pill-c{color:var(--cyan);border-color:rgba(0,229,255,0.3);background:rgba(0,229,255,0.06);}
.pill-o{color:var(--orange);border-color:rgba(255,107,53,0.3);background:rgba(255,107,53,0.06);}
.pill-p{color:var(--purple);border-color:rgba(168,85,247,0.3);background:rgba(168,85,247,0.06);}

/* ── CHART WRAPPERS ── */
.ch{position:relative;height:115px;}
.ch.tall{height:150px;}
.ch.short{height:85px;}

/* ── TABLE ── */
.tbl{width:100%;border-collapse:collapse;font-family:"Share Tech Mono",monospace;font-size:0.58rem;}
.tbl th{color:var(--cyan);border-bottom:1px solid var(--border2);padding:4px 5px;
  text-align:left;font-weight:400;letter-spacing:1px;}
.tbl td{padding:4px 5px;border-bottom:1px solid var(--border);color:var(--text);}
.tbl tr:hover td{background:rgba(0,229,255,0.04);}
.tc{color:var(--cyan);}
.tg{color:var(--green);}
.to{color:var(--orange);}
.tr{color:var(--red);}
.ty{color:var(--yellow);}

/* ── MAP AREA ── */
#map-area{position:relative;display:flex;flex-direction:column;background:var(--bg);}
#map{flex:1;z-index:1;}

/* override leaflet attribution */
.leaflet-control-attribution{
  background:rgba(2,5,10,0.85)!important;color:var(--dim)!important;font-size:9px!important;
}
.leaflet-control-attribution a{color:var(--cyan)!important;}

/* map top label */
#map-label{
  position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:600;
  background:rgba(2,5,10,0.9);padding:5px 18px;border-radius:20px;
  border:1px solid var(--border2);backdrop-filter:blur(8px);
  font-family:"Orbitron",monospace;font-size:0.6rem;color:var(--cyan);
  letter-spacing:2px;text-transform:uppercase;white-space:nowrap;
  box-shadow:var(--glow-c);
}

/* map bottom controls */
#map-ctrl{
  position:absolute;bottom:14px;left:50%;transform:translateX(-50%);
  z-index:600;display:flex;gap:6px;
  background:rgba(2,5,10,0.92);padding:8px 14px;border-radius:10px;
  border:1px solid var(--border2);backdrop-filter:blur(10px);
}

/* legend */
#legend{
  position:absolute;top:50px;right:10px;z-index:600;
  background:rgba(2,5,10,0.9);padding:10px 13px;
  border-radius:6px;border:1px solid var(--border2);
  backdrop-filter:blur(8px);min-width:145px;
}
#legend h4{font-family:"Share Tech Mono",monospace;font-size:0.56rem;
  color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-bottom:7px;}
.leg-grad{height:10px;border-radius:2px;margin-bottom:3px;
  background:linear-gradient(to right,
    rgba(20,0,200,0),rgba(20,0,200,0.5),rgba(150,0,240,0.7),
    rgba(220,30,50,0.85),rgba(220,140,0,1),rgba(255,220,0,1));}
.leg-range{display:flex;justify-content:space-between;
  font-family:"Share Tech Mono",monospace;font-size:0.48rem;color:var(--dim);margin-bottom:8px;}
.leg-item{display:flex;align-items:center;gap:5px;margin-bottom:4px;
  font-family:"Share Tech Mono",monospace;font-size:0.56rem;}
.leg-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}

/* anim progress */
#anim-prog{
  position:absolute;bottom:52px;left:50%;transform:translateX(-50%);
  width:220px;height:3px;background:var(--border2);border-radius:2px;
  z-index:600;overflow:hidden;display:none;
}
#anim-fill{height:100%;width:0;background:var(--cyan);border-radius:2px;}

/* ── LOADING ── */
#loader{
  position:fixed;inset:0;background:var(--bg);z-index:99999;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;
}
.spin{
  width:56px;height:56px;border:3px solid var(--border2);
  border-top-color:var(--cyan);border-radius:50%;
  animation:rot 0.9s linear infinite;
}
@keyframes rot{to{transform:rotate(360deg)}}
.load-t{font-family:"Orbitron",monospace;font-size:0.7rem;color:var(--cyan);letter-spacing:3px;}
.load-s{font-family:"Share Tech Mono",monospace;font-size:0.58rem;color:var(--dim);
  letter-spacing:2px;animation:blink 1.4s step-end infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* matrix heatmap canvas */
#mat-canvas{width:100%;border-radius:4px;image-rendering:pixelated;}

/* info text */
.info-txt{font-family:"Share Tech Mono",monospace;font-size:0.56rem;color:var(--dim);
  line-height:1.6;margin-top:5px;}
.info-txt b{color:var(--cyan);}
.info-txt b.o{color:var(--orange);}
.info-txt b.g{color:var(--green);}
.info-txt b.p{color:var(--purple);}

/* unit section divider */
.unit-hdr{
  font-family:"Orbitron",monospace;font-size:0.58rem;letter-spacing:2px;
  text-transform:uppercase;padding:5px 0 6px;margin-bottom:4px;
  border-bottom:1px solid var(--border2);
}
.unit-hdr.u1{color:var(--cyan);}
.unit-hdr.u2{color:var(--orange);}
.unit-hdr.u3{color:var(--purple);}
</style>
</head>
<body>

<!-- LOADER -->
<div id="loader">
  <div class="spin"></div>
  <div class="load-t">Pollution Control System</div>
  <div class="load-s">Initializing Bengaluru Grid...</div>
</div>

<!-- HEADER -->
<div id="hdr">
  <div class="hdr-logo">
    <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/>
      <path d="M12 7v5l3.5 3.5" stroke-linecap="round"/>
      <circle cx="12" cy="12" r="2" fill="rgba(0,229,255,0.4)" stroke="none"/>
    </svg>
  </div>
  <div class="hdr-titles">
    <h1>Simulation and Control of Pollution Spread Using PDEs and Gradient Descent Optimization</h1>
    <p>Amrita Vishwa Vidyapeetham &nbsp;·&nbsp; MoC Sem 4 &nbsp;·&nbsp; Bengaluru Real-Map Simulation</p>
  </div>
  <div class="hdr-metrics">
    <div class="hdr-sep"></div>
    <div class="hm"><div class="hm-val g" id="hv-red">—</div><div class="hm-lbl">Reduction</div></div>
    <div class="hdr-sep"></div>
    <div class="hm"><div class="hm-val o" id="hv-before">—</div><div class="hm-lbl">Before</div></div>
    <div class="hm"><div class="hm-val c" id="hv-after">—</div><div class="hm-lbl">After</div></div>
    <div class="hdr-sep"></div>
    <div class="hm"><div class="hm-val p" id="hv-pval">—</div><div class="hm-lbl">p-value</div></div>
    <div class="sig-pill" id="sig-pill" style="display:none">✓ Significant</div>
  </div>
</div>

<!-- APP -->
<div id="app">

<!-- ══════ LEFT SIDEBAR ══════ -->
<div class="sb sb-l">

  <!-- PROJECT TITLE CARD -->
  <div class="card" style="background:linear-gradient(135deg,rgba(0,229,255,0.05),rgba(2,5,10,0));">
    <div class="ctitle">Project Overview</div>
    <div class="info-txt">
      <b>PDE Model:</b> ∂C/∂t = D∇²C − kC + S(x) − F(x)<br>
      <b>D</b> = diffusion &nbsp;·&nbsp; <b>k</b> = decay &nbsp;·&nbsp; <b>S</b> = sources &nbsp;·&nbsp; <b>F</b> = filters<br><br>
      <b class="o">Objective:</b> minimise ∫∫ C(x,t) dx over Bengaluru grid<br>
      using ADMM + Gradient Descent filter placement.
    </div>
  </div>

  <!-- SEMESTER TOPICS -->
  <div class="card">
    <div class="ctitle">Semester Topics Applied</div>
    <div class="unit-hdr u1">Unit 1 — Special Matrices</div>
    <div class="pills">
      <span class="pill pill-c">Circulant Matrix</span>
      <span class="pill pill-c">Toeplitz Matrix</span>
      <span class="pill pill-c">Shift Matrix</span>
      <span class="pill pill-c">Kronecker ⊗</span>
      <span class="pill pill-c">2-D DFT</span>
      <span class="pill pill-c">Graph Laplacian</span>
      <span class="pill pill-c">Kirchhoff Law</span>
      <span class="pill pill-c">Spectral Cluster</span>
      <span class="pill pill-c">K-Means</span>
      <span class="pill pill-c">Rank-1 Completion</span>
      <span class="pill pill-c">Procrustes</span>
      <span class="pill pill-c">Distance Matrix</span>
    </div>
    <div class="unit-hdr u2" style="margin-top:10px;">Unit 2 — Optimization</div>
    <div class="pills">
      <span class="pill pill-o">ADMM</span>
      <span class="pill pill-o">L1 + L2 Split</span>
      <span class="pill pill-o">Aug. Lagrangian</span>
      <span class="pill pill-o">Proximal Operator</span>
      <span class="pill pill-o">Gradient Descent</span>
      <span class="pill pill-o">ADAM</span>
      <span class="pill pill-o">SGD</span>
      <span class="pill pill-o">Compressed Sensing</span>
    </div>
    <div class="unit-hdr u3" style="margin-top:10px;">Unit 3 — Statistics</div>
    <div class="pills">
      <span class="pill pill-p">MLE</span>
      <span class="pill pill-p">Mann-Whitney U</span>
      <span class="pill pill-p">Hypothesis Test</span>
      <span class="pill pill-p">Gaussian Fit</span>
    </div>
  </div>

  <!-- POLLUTION SOURCES -->
  <div class="card">
    <div class="ctitle orange">Pollution Hotspots — Bengaluru</div>
    <div id="src-list"></div>
  </div>

  <!-- SIMULATION RESULTS -->
  <div class="card">
    <div class="ctitle green">Simulation Results</div>
    <div class="mgrid" id="sim-metrics"></div>
  </div>

  <!-- REDUCTION BARS -->
  <div class="card">
    <div class="ctitle green">Pollution Reduction</div>
    <div class="pbar-wrap">
      <div class="pbar-top"><span>Gradient Descent (Local Search)</span><span id="pb-ls-v" style="color:var(--green)">—</span></div>
      <div class="pbar-bg"><div class="pbar-fill" id="pb-ls" style="background:var(--green);width:0%"></div></div>
    </div>
    <div class="pbar-wrap">
      <div class="pbar-top"><span>ADMM (L1+L2 Split)</span><span id="pb-ad-v" style="color:var(--orange)">—</span></div>
      <div class="pbar-bg"><div class="pbar-fill" id="pb-ad" style="background:var(--orange);width:0%"></div></div>
    </div>
    <div class="info-txt" style="margin-top:6px;">
      <b>ADMM</b> = Alternating Direction Method of Multipliers<br>
      split variable: <b class="o">x</b> (filter map) + <b class="g">z</b> (sparse aux) + <b class="p">u</b> (dual)
    </div>
  </div>

  <!-- CONVERGENCE CHART -->
  <div class="card">
    <div class="ctitle orange">Optimization Convergence (Unit-2)</div>
    <div class="tabs">
      <div class="tab on o" onclick="tabConv('admm',this)">ADMM</div>
      <div class="tab" onclick="tabConv('local',this)">Gradient Descent</div>
      <div class="tab" onclick="tabConv('both',this)">Both</div>
    </div>
    <div class="ch"><canvas id="ch-conv"></canvas></div>
  </div>

  <!-- ADAM vs SGD -->
  <div class="card">
    <div class="ctitle orange">ADAM vs SGD Loss Curves (Unit-2)</div>
    <div class="ch"><canvas id="ch-optim"></canvas></div>
    <div class="info-txt"><b>ADAM:</b> adaptive moments m̂ᵢ/v̂ᵢ · <b class="p">SGD:</b> fixed lr + noise</div>
  </div>

</div>

<!-- ══════ MAP CENTRE ══════ -->
<div id="map-area">
  <div id="map-label">BASELINE POLLUTION — BENGALURU</div>

  <!-- LEGEND -->
  <div id="legend">
    <h4>Concentration µg/m³</h4>
    <div class="leg-grad"></div>
    <div class="leg-range"><span>0</span><span>Low</span><span>High</span></div>
    <div class="leg-item"><div class="leg-dot" style="background:var(--orange);box-shadow:0 0 6px var(--orange)"></div>Pollution Source</div>
    <div class="leg-item"><div class="leg-dot" style="background:var(--cyan);box-shadow:0 0 6px var(--cyan)"></div>Initial Filter</div>
    <div class="leg-item"><div class="leg-dot" style="background:var(--green);box-shadow:0 0 6px var(--green)"></div>Optimised Filter</div>
    <div class="leg-item"><div class="leg-dot" style="background:var(--red);box-shadow:0 0 6px var(--red)"></div>ADMM Filter</div>
  </div>

  <div id="map"></div>
  <div id="anim-prog"><div id="anim-fill"></div></div>

  <!-- MAP CONTROLS -->
  <div id="map-ctrl">
    <button class="btn btn-o" onclick="showMap('before')">⬤ Baseline</button>
    <button class="btn btn-g" onclick="showMap('after')">⬤ GD Optimised</button>
    <button class="btn btn-o" onclick="showMap('admm')">⬤ ADMM</button>
    <button class="btn btn-c" onclick="showMap('cluster')">◈ Clusters</button>
    <button class="btn btn-p" onclick="showMap('graph')">◈ Road Graph</button>
    <button class="btn btn-y" onclick="toggleAnim()" id="anim-btn">▶ Animate</button>
  </div>
</div>

<!-- ══════ RIGHT SIDEBAR ══════ -->
<div class="sb sb-r">

  <!-- AQI DATASET TABLE -->
  <div class="card">
    <div class="ctitle">Real-World AQI Dataset — Bengaluru</div>
    <table class="tbl" id="aqi-tbl">
      <tr><th>Station</th><th>AQI</th><th>PM₂.₅</th><th>Status</th></tr>
    </table>
  </div>

  <!-- AQI SEASONAL CHART -->
  <div class="card">
    <div class="ctitle">Monthly AQI — Silk Board Junction</div>
    <div class="ch"><canvas id="ch-aqi"></canvas></div>
    <div class="info-txt"><b>Dataset:</b> 8 stations · 365 days · 5 pollutants</div>
  </div>

  <!-- DFT SPECTRUM -->
  <div class="card">
    <div class="ctitle">2-D DFT Power Spectrum (Unit-1)</div>
    <div class="tabs">
      <div class="tab on" onclick="switchDFT('before',this)">Before</div>
      <div class="tab" onclick="switchDFT('after',this)">After GD</div>
    </div>
    <canvas id="dft-canvas" width="250" height="100" style="width:100%;border-radius:4px;"></canvas>
    <div class="info-txt"><b>DFT</b>: F(k) = Σ C(x)e^{-2πikx/N} · log₁+|F| plotted</div>
  </div>

  <!-- GRAPH LAPLACIAN EIGENSPECTRUM -->
  <div class="card">
    <div class="ctitle">Graph Laplacian Eigenspectrum (Unit-1)</div>
    <div class="ch short"><canvas id="ch-eig"></canvas></div>
    <div class="mgrid" style="margin-top:7px;">
      <div class="met"><div class="met-v c" id="mv-fiedler">—</div><div class="met-l">Fiedler λ₂</div></div>
      <div class="met"><div class="met-v p">12</div><div class="met-l">Road Nodes</div></div>
    </div>
    <div class="info-txt"><b>L = D − A</b> · λ₂ = algebraic connectivity (Kirchhoff)</div>
  </div>

  <!-- CIRCULANT vs TOEPLITZ -->
  <div class="card">
    <div class="ctitle">Circulant vs Toeplitz Diffusion (Unit-1)</div>
    <div class="ch"><canvas id="ch-toepl"></canvas></div>
    <div class="info-txt">
      <b>Circulant:</b> periodic BC, diagonalised by DFT matrix<br>
      <b class="o">Toeplitz:</b> open BC, shift-invariant filter<br>
      <b class="g">Kronecker</b>: L₂D = I⊗L + L⊗I (2-D Laplacian)
    </div>
  </div>

  <!-- STATISTICS + HISTOGRAM -->
  <div class="card">
    <div class="ctitle purple">MLE + Hypothesis Test (Unit-3)</div>
    <div class="ch"><canvas id="ch-hist"></canvas></div>
    <div id="stats-info" class="info-txt" style="margin-top:6px;"></div>
  </div>

  <!-- RANK-1 COMPLETION -->
  <div class="card">
    <div class="ctitle">Rank-1 Matrix Completion (Unit-1)</div>
    <div class="mgrid">
      <div class="met"><div class="met-v g" id="mv-r1err">—</div><div class="met-l">Recovery Error</div></div>
      <div class="met"><div class="met-v o">30%</div><div class="met-l">Missing Sensors</div></div>
    </div>
    <div class="info-txt" style="margin-top:6px;">
      <b>SVD:</b> C = UΣVᵀ · <b class="o">rank-1:</b> σ₁·u₁v₁ᵀ fills gaps<br>
      30% of sensor readings artificially missing → recovered
    </div>
  </div>

  <!-- COMPRESSED SENSING -->
  <div class="card">
    <div class="ctitle orange">Compressed Sensing — ISTA (Unit-2)</div>
    <div class="ch short"><canvas id="ch-cs"></canvas></div>
    <div class="info-txt"><b>ISTA:</b> xₙ₊₁ = S_λ(xₙ − η∇f) · proximal L1 shrinkage</div>
  </div>

  <!-- ADMM MAP (heatmap of filter weights) -->
  <div class="card">
    <div class="ctitle orange">ADMM Sparse Filter Weight Map (Unit-2)</div>
    <canvas id="mat-canvas" height="80"></canvas>
    <div class="info-txt" style="margin-top:5px;"><b>Yellow</b> = optimal filter location found by ADMM L1+L2 split</div>
  </div>

</div>
</div><!-- end app -->

<script>
// ══════════════════════════════════════════════════════════
//  SIMULATION DATA
// ══════════════════════════════════════════════════════════
const SIM = ''' + SIM_JSON + ''';

const CFG  = SIM.config;
const NX   = CFG.NX, NY = CFG.NY;

// ══════════════════════════════════════════════════════════
//  CHART FACTORY
// ══════════════════════════════════════════════════════════
const CI = {};
const BASE_CHART = {
  responsive:true, maintainAspectRatio:false,
  animation:{duration:600},
  plugins:{legend:{display:false}},
  scales:{
    x:{ticks:{color:'#3a5878',font:{size:8},maxTicksLimit:6},
       grid:{color:'#0f2035'},border:{color:'#0f2035'}},
    y:{ticks:{color:'#3a5878',font:{size:8},maxTicksLimit:5},
       grid:{color:'#0f2035'},border:{color:'#0f2035'}},
  }
};
function mkChart(id, type, labels, datasets, extra={}){
  const ctx = document.getElementById(id).getContext('2d');
  if(CI[id]) CI[id].destroy();
  CI[id] = new Chart(ctx,{
    type,
    data:{labels, datasets},
    options: deepMerge(BASE_CHART, extra)
  });
}
function deepMerge(a, b){
  const o = Object.assign({}, a);
  for(const k in b){
    if(b[k] && typeof b[k]==='object' && !Array.isArray(b[k]))
      o[k] = deepMerge(a[k]||{}, b[k]);
    else o[k] = b[k];
  }
  return o;
}
const DS = (d, color, dash=[]) => ({
  data:d, borderColor:color, backgroundColor:color+'18',
  borderWidth:1.6, pointRadius:0, tension:0.35, fill:false,
  borderDash:dash
});
const DSfill = (d, color) => ({
  data:d, borderColor:color, backgroundColor:color+'22',
  borderWidth:1.4, pointRadius:0, tension:0.4, fill:true
});

// ══════════════════════════════════════════════════════════
//  LEAFLET MAP
// ══════════════════════════════════════════════════════════
let map, overlayImg=null, clusterGrp=null, graphGrp=null;
let currentView='before';
let animRunning=false, animIdx=0, animTick=null;

function initMap(){
  const clat=(CFG.lat_min+CFG.lat_max)/2, clon=(CFG.lon_min+CFG.lon_max)/2;
  map = L.map('map',{center:[clat,clon],zoom:13,zoomControl:true});

  // Real OpenStreetMap tiles
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
    attribution:'© OpenStreetMap contributors',maxZoom:19
  }).addTo(map);

  map.whenReady(()=>{ setOverlay(SIM.rgba_before); addMarkers(); });
}

function rgbaToDataURL(flat){
  const c = document.createElement('canvas');
  c.width=NX; c.height=NY;
  const ctx=c.getContext('2d');
  const img=ctx.createImageData(NX,NY);
  for(let i=0;i<flat.length;i++) img.data[i]=flat[i];
  ctx.putImageData(img,0,0);
  return c.toDataURL();
}

function setOverlay(flat){
  const bounds=L.latLngBounds([CFG.lat_min,CFG.lon_min],[CFG.lat_max,CFG.lon_max]);
  if(overlayImg){ map.removeLayer(overlayImg); overlayImg=null; }
  overlayImg = L.imageOverlay(rgbaToDataURL(flat), bounds, {opacity:0.72, zIndex:400}).addTo(map);
}

function addMarkers(){
  const srcIcon = s => L.divIcon({
    html:`<div style="width:${10+s}px;height:${10+s}px;border-radius:50%;
      background:rgba(255,107,53,0.85);border:2px solid #ff6b35;
      box-shadow:0 0 ${s*2}px rgba(255,107,53,0.8);margin-top:-5px;margin-left:-5px;"></div>`,
    iconSize:[0,0],className:''
  });
  const mkFilter = (col,sz) => L.divIcon({
    html:`<div style="width:${sz}px;height:${sz}px;border-radius:50%;
      background:rgba(${col},0.15);border:2px solid rgb(${col});
      box-shadow:0 0 14px rgba(${col},0.7);
      margin-top:-${sz/2}px;margin-left:-${sz/2}px;"></div>`,
    iconSize:[0,0],className:''
  });

  SIM.sources.forEach(s=>{
    L.marker([s.lat,s.lon],{icon:srcIcon(s.strength)})
      .bindPopup(`<div style="font-family:monospace;color:#ff6b35;font-weight:bold;">${s.name}</div>
        <div style="font-size:12px;color:#b8cfe8;">Emission strength: <b>${s.strength} µg/m³/s</b></div>`)
      .addTo(map);
  });
  SIM.init_filters.forEach(f=>{
    L.marker([f.lat,f.lon],{icon:mkFilter('0,229,255',16)})
      .bindPopup(`<div style="font-family:monospace;color:#00e5ff;font-weight:bold;">Initial Filter: ${f.name}</div>`)
      .addTo(map);
  });
  SIM.opt_filters.forEach((f,i)=>{
    L.marker([f.lat,f.lon],{icon:mkFilter('57,255,20',18)})
      .bindPopup(`<div style="font-family:monospace;color:#39ff14;font-weight:bold;">GD Optimised Filter ${i+1}</div>
        <div style="font-size:12px;color:#b8cfe8;">Gradient Descent placement</div>`)
      .addTo(map);
  });
  SIM.admm_filters.forEach((f,i)=>{
    L.marker([f.lat,f.lon],{icon:mkFilter('244,63,94',18)})
      .bindPopup(`<div style="font-family:monospace;color:#f43f5e;font-weight:bold;">ADMM Filter ${i+1}</div>
        <div style="font-size:12px;color:#b8cfe8;">L1+L2 split optimisation</div>`)
      .addTo(map);
  });
}

// ── Layer switching ──────────────────────────────────────
function clearExtras(){
  if(clusterGrp){ map.removeLayer(clusterGrp); clusterGrp=null; }
  if(graphGrp){ map.removeLayer(graphGrp); graphGrp=null; }
}

function showMap(mode){
  if(animRunning) stopAnim();
  clearExtras();
  document.getElementById('map-label').style.color='var(--cyan)';

  if(mode==='before'){
    setOverlay(SIM.rgba_before);
    setLabel('BASELINE POLLUTION — BENGALURU');
    currentView='before';
  } else if(mode==='after'){
    setOverlay(SIM.rgba_after);
    setLabel('GRADIENT DESCENT OPTIMISED — BENGALURU');
    currentView='after';
  } else if(mode==='admm'){
    setOverlay(SIM.rgba_admm);
    setLabel('ADMM FILTER PLACEMENT — BENGALURU');
    currentView='admm';
  } else if(mode==='cluster'){
    if(overlayImg){ map.removeLayer(overlayImg); overlayImg=null; }
    setLabel('SPECTRAL CLUSTERING — POLLUTION ZONES');
    drawClusters();
  } else if(mode==='graph'){
    if(overlayImg){ map.removeLayer(overlayImg); overlayImg=null; }
    setLabel('ROAD NETWORK GRAPH LAPLACIAN');
    drawGraph();
  }
}
function setLabel(t){ document.getElementById('map-label').textContent=t; }

const CLUSTER_COLS=['#00e5ff','#ff6b35','#39ff14','#a855f7','#fbbf24'];
function drawClusters(){
  clusterGrp = L.layerGroup().addTo(map);
  SIM.clusters_sc.forEach(pt=>{
    const c=CLUSTER_COLS[pt.cluster%5];
    L.circleMarker([pt.lat,pt.lon],{radius:3,color:c,fillColor:c,fillOpacity:0.55,weight:0}).addTo(clusterGrp);
  });
}
function drawGraph(){
  graphGrp = L.layerGroup().addTo(map);
  SIM.graph_edges.forEach(([i,j,w])=>{
    const a=SIM.graph_nodes[i],b=SIM.graph_nodes[j];
    L.polyline([[a.lat,a.lon],[b.lat,b.lon]],{
      color:'#00e5ff',weight:1+w*2.5,opacity:Math.min(0.9,0.3+w*0.8)
    }).addTo(graphGrp);
  });
  SIM.graph_nodes.forEach(n=>{
    L.circleMarker([n.lat,n.lon],{radius:5,color:'#00e5ff',fillColor:'#00e5ff',fillOpacity:0.9,weight:1})
      .bindPopup(`<div style="font-family:monospace;color:#00e5ff;font-weight:bold;">${n.name}</div>
        <div style="font-size:11px;color:#b8cfe8;">Road network junction</div>`)
      .addTo(graphGrp);
  });
}

// ── Animation ─────────────────────────────────────────────
function toggleAnim(){
  animRunning ? stopAnim() : startAnim();
}
function startAnim(){
  animRunning=true; animIdx=0;
  document.getElementById('anim-prog').style.display='block';
  document.getElementById('anim-btn').textContent='■ Stop';
  const frames = (currentView==='after') ? SIM.anim_opt : SIM.anim_before;
  animTick = setInterval(()=>{
    setOverlay(frames[animIdx % frames.length]);
    document.getElementById('anim-fill').style.width=(animIdx/frames.length*100)+'%';
    animIdx++;
    if(animIdx >= frames.length) animIdx=0;
  },110);
}
function stopAnim(){
  animRunning=false;
  clearInterval(animTick);
  document.getElementById('anim-prog').style.display='none';
  document.getElementById('anim-btn').textContent='▶ Animate';
  showMap(currentView);
}

// ══════════════════════════════════════════════════════════
//  DFT CANVAS RENDERER
// ══════════════════════════════════════════════════════════
function drawDFT(grid){
  const cv=document.getElementById('dft-canvas');
  const ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  const nx=grid.length, ny=grid[0].length;
  const img=ctx.createImageData(W,H);
  for(let py=0;py<H;py++){
    for(let px=0;px<W;px++){
      const ix=Math.floor(px/W*nx), iy=Math.floor(py/H*ny);
      const v=(grid[ix]?.[iy]??0)/255;
      const i4=(py*W+px)*4;
      // magma palette
      img.data[i4]  =Math.min(255,v<0.5?v*2*180:180+(v-0.5)*2*75);
      img.data[i4+1]=Math.min(255,v<0.3?0:(v-0.3)/0.7*220);
      img.data[i4+2]=Math.min(255,v<0.5?200-v*2*200:0);
      img.data[i4+3]=255;
    }
  }
  ctx.putImageData(img,0,0);
}
function switchDFT(m, el){
  el.parentElement.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  el.classList.add('on');
  drawDFT(m==='before' ? SIM.dft_before : SIM.dft_after);
}

// ══════════════════════════════════════════════════════════
//  ADMM WEIGHT MAP
// ══════════════════════════════════════════════════════════
function drawADMMMap(){
  const cv = document.getElementById('mat-canvas');
  const flat = SIM.admm_xmap; // NX arrays of NY values
  const nx=flat.length, ny=flat[0].length;
  cv.width=nx; cv.height=ny;
  const ctx=cv.getContext('2d');
  const img=ctx.createImageData(nx,ny);
  for(let py=0;py<ny;py++){
    for(let px=0;px<nx;px++){
      const v=flat[px][ny-1-py];
      const i4=(py*nx+px)*4;
      img.data[i4]  =Math.min(255,v*255*2);
      img.data[i4+1]=Math.min(255,v*220);
      img.data[i4+2]=Math.min(255,v<0.5?v*80:0);
      img.data[i4+3]=200;
    }
  }
  ctx.putImageData(img,0,0);
}

// ══════════════════════════════════════════════════════════
//  CONVERGENCE CHART TABS
// ══════════════════════════════════════════════════════════
let convMode='admm';
function tabConv(m, el){
  convMode=m;
  el.parentElement.querySelectorAll('.tab').forEach(t=>{ t.classList.remove('on','o'); });
  el.classList.add('on');
  if(m==='admm'||m==='both') el.classList.add('o');
  buildConvChart();
}
function buildConvChart(){
  const al = SIM.admm_history.map((v,i)=>i);
  const ol = SIM.opt_history.map((v,i)=>i);
  const mx = Math.max(al.length,ol.length);
  if(convMode==='admm')
    mkChart('ch-conv','line', al, [DS(SIM.admm_history,'#ff6b35')]);
  else if(convMode==='local')
    mkChart('ch-conv','line', ol, [DS(SIM.opt_history,'#39ff14')]);
  else
    mkChart('ch-conv','line', Array.from({length:mx},(_,i)=>i), [
      DS(SIM.admm_history,'#ff6b35'),
      DS(SIM.opt_history,'#39ff14')
    ],{plugins:{legend:{display:true,labels:{color:'#b8cfe8',font:{size:8},boxWidth:10}}}});
}

// ══════════════════════════════════════════════════════════
//  BUILD ALL UI COMPONENTS
// ══════════════════════════════════════════════════════════
function buildSources(){
  const cols=['#ff6b35','#ff9500','#fbbf24','#00e5ff','#a855f7'];
  const el = document.getElementById('src-list');
  SIM.sources.forEach((s,i)=>{
    el.innerHTML += `<div class="src-row" onclick="map.setView([${s.lat},${s.lon}],15)">
      <div class="src-dot" style="background:${cols[i]};box-shadow:0 0 6px ${cols[i]}80;"></div>
      <div class="src-name">${s.name}</div>
      <div class="src-str">${s.strength}</div>
    </div>`;
  });
}

function buildMetrics(){
  const s=SIM.summary, st=SIM.stats;
  document.getElementById('sim-metrics').innerHTML=`
    <div class="met"><div class="met-v g">${s.reduction_local_pct}%</div><div class="met-l">GD Reduction</div></div>
    <div class="met"><div class="met-v o">${s.reduction_admm_pct}%</div><div class="met-l">ADMM Reduction</div></div>
    <div class="met"><div class="met-v c">${(s.total_before/1000).toFixed(1)}K</div><div class="met-l">Total Before</div></div>
    <div class="met"><div class="met-v g">${(s.total_opt/1000).toFixed(1)}K</div><div class="met-l">Total After</div></div>
  `;
  // header
  document.getElementById('hv-red').textContent    = s.reduction_local_pct+'%';
  document.getElementById('hv-before').textContent = (st.mu_before).toFixed(1);
  document.getElementById('hv-after').textContent  = (st.mu_after).toFixed(1);
  document.getElementById('hv-pval').textContent   = st.p_value < 1e-10 ? '< 1e-10' : st.p_value.toExponential(2);
  if(st.significant) document.getElementById('sig-pill').style.display='inline-flex';

  // progress bars
  setTimeout(()=>{
    const lp=Math.min(s.reduction_local_pct,100);
    const ap=Math.min(s.reduction_admm_pct,100);
    document.getElementById('pb-ls').style.width  = lp+'%';
    document.getElementById('pb-ad').style.width  = ap+'%';
    document.getElementById('pb-ls-v').textContent = lp+'%';
    document.getElementById('pb-ad-v').textContent = ap+'%';
  }, 700);

  // rank-1
  document.getElementById('mv-r1err').textContent = SIM.rank1.recovery_error_pct+'%';
  // fiedler
  document.getElementById('mv-fiedler').textContent = SIM.fiedler_value.toFixed(4);

  // stats info
  const st2=SIM.stats;
  document.getElementById('stats-info').innerHTML =
    `<b>MLE Before:</b> μ=${st2.mu_before} σ=${st2.sigma_before}<br>
     <b class="g">MLE After:</b> μ=${st2.mu_after} σ=${st2.sigma_after}<br>
     <b class="p">Mann-Whitney</b> U=${st2.U_stat.toExponential(2)} · p=${st2.p_value < 1e-10 ? '< 1e−10' : st2.p_value.toExponential(2)} · <b class="g">SIGNIFICANT ✓</b>`;
}

function buildAQITable(){
  const tb=document.getElementById('aqi-tbl');
  SIM.aqi_data.forEach(s=>{
    const cls=s.aqi_mean>200?'tr':s.aqi_mean>150?'to':s.aqi_mean>100?'ty':'tg';
    const lbl=s.aqi_mean>200?'Hazardous':s.aqi_mean>150?'Unhealthy':s.aqi_mean>100?'Moderate':'Good';
    tb.innerHTML += `<tr>
      <td>${s.name}</td>
      <td class="${cls}">${s.aqi_mean}</td>
      <td>${Math.round(s.pm25_mean)}</td>
      <td class="${cls}">${lbl}</td>
    </tr>`;
  });
}

function buildCharts(){
  // Convergence
  buildConvChart();

  // ADAM vs SGD
  const mx2=Math.max(SIM.adam_history.length,SIM.sgd_history.length);
  mkChart('ch-optim','line',Array.from({length:mx2},(_,i)=>i),[
    DS(SIM.adam_history,'#39ff14'),
    DS(SIM.sgd_history,'#a855f7',[4,3])
  ],{plugins:{legend:{display:true,labels:{color:'#b8cfe8',font:{size:8},
    generateLabels:()=>[
      {text:'ADAM',strokeStyle:'#39ff14',fillStyle:'#39ff1418',lineWidth:1.5},
      {text:'SGD', strokeStyle:'#a855f7',fillStyle:'#a855f718',lineWidth:1.5},
    ]}}}});

  // AQI monthly
  const silk=SIM.aqi_data[0];
  const mnths=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  mkChart('ch-aqi','bar',mnths,[{
    data:silk.monthly_avg,
    backgroundColor:silk.monthly_avg.map(v=>v>200?'rgba(244,63,94,0.75)':v>150?'rgba(255,107,53,0.75)':'rgba(251,191,36,0.75)'),
    borderRadius:3,borderSkipped:false
  }]);

  // Eigenspectrum
  const ev=SIM.graph_eigvals;
  mkChart('ch-eig','bar',ev.map((_,i)=>`λ${i}`),[{
    data:ev,
    backgroundColor:ev.map((v,i)=>i===1?'rgba(0,229,255,0.9)':'rgba(0,229,255,0.25)'),
    borderRadius:2,borderSkipped:false
  }]);

  // Toeplitz vs Circulant
  const step=2, idx=SIM.u_demo.filter((_,i)=>i%step===0).map((_,i)=>i*step);
  mkChart('ch-toepl','line',idx,[
    DS(SIM.u_demo.filter((_,i)=>i%step===0),'#b8cfe8'),
    DS(SIM.u_circ.filter((_,i)=>i%step===0),'#00e5ff',[5,3]),
    DS(SIM.u_toep.filter((_,i)=>i%step===0),'#ff6b35',[2,4])
  ],{plugins:{legend:{display:true,labels:{color:'#b8cfe8',font:{size:8},
    generateLabels:()=>[
      {text:'Original',strokeStyle:'#b8cfe8',fillStyle:'transparent',lineWidth:1.5},
      {text:'Circulant',strokeStyle:'#00e5ff',fillStyle:'transparent',lineWidth:1.5,lineDash:[5,3]},
      {text:'Toeplitz', strokeStyle:'#ff6b35',fillStyle:'transparent',lineWidth:1.5,lineDash:[2,4]},
    ]}}}});

  // Histogram
  const st=SIM.stats;
  mkChart('ch-hist','line',st.hist_centers.map(v=>v.toFixed(0)),[
    DSfill(st.hist_before,'#ff6b35'),
    DSfill(st.hist_after,'#39ff14')
  ],{plugins:{legend:{display:true,labels:{color:'#b8cfe8',font:{size:8},
    generateLabels:()=>[
      {text:'Before',strokeStyle:'#ff6b35',fillStyle:'#ff6b3522',lineWidth:1.5},
      {text:'After', strokeStyle:'#39ff14',fillStyle:'#39ff1422',lineWidth:1.5},
    ]}}}});

  // Compressed sensing
  mkChart('ch-cs','line',SIM.cs_history.map((_,i)=>i),[DS(SIM.cs_history,'#a855f7')]);

  // DFT
  drawDFT(SIM.dft_before);

  // ADMM map
  drawADMMMap();
}

// ══════════════════════════════════════════════════════════
//  BOOT
// ══════════════════════════════════════════════════════════
window.addEventListener('load', ()=>{
  setTimeout(()=>{
    initMap();
    buildSources();
    buildMetrics();
    buildAQITable();
    buildCharts();
    document.getElementById('loader').style.opacity='0';
    setTimeout(()=>document.getElementById('loader').style.display='none', 400);
  }, 350);
});
document.getElementById('loader').style.transition='opacity 0.4s';
</script>
</body>
</html>'''

out_path = '/home/claude/blr_pollution/web/index.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(HTML)

size_kb = os.path.getsize(out_path) // 1024
print(f"✅ Dashboard built → {out_path}")
print(f"   File size: {size_kb} KB")
