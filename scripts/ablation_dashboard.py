"""
ablation_dashboard.py — interactive per-feature blend explorer.

Layout
------
  Row 1  SOURCE | FINAL COMBINED | ORIGINAL BLENDED   (3-col, large)
  Row 2  per-step post-processing previews             (5-col, medium)
  Row 3  per-feature ablation with +/- alpha controls
  Footer RUN   SAVE (writes alphas to JSON)

MoMask inference runs once on startup.  RUN reruns only the fast
autoregressive reconstruction + post-processing.

Usage
-----
    python scripts/ablation_dashboard.py \\
        --jobs_file jobs.json \\
        --job_idx 0 \\
        --alpha 0.5 \\
        --out_dir dashboard_outputs \\
        --gpu_id 0 \\
        --port 5050
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from flask import Flask, jsonify, request, send_file
except ImportError:
    print("Flask is required:  pip install flask")
    sys.exit(1)

import torch

from gen_t2m import load_res_model, load_trans_model, load_vq_model
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.paramUtil import t2m_kinematic_chain
from utils.plot_script import plot_3d_motion
from visualization.joints2bvh import Joint2BVHConvertor

from semantic_spectrum.blend import FEATURE_NAMES, _IDX_FREQ, _IDX_FREQ_MAG
from semantic_spectrum.pipeline import ZoneBlendPipeline
from semantic_spectrum.postprocess import STEPS as PP_STEPS, STEP_KEYS, apply_pipeline

# ── Constants ────────────────────────────────────────────────────────────────

ABLATE_SETS: list[tuple[set[int], str]] = (
    [({i}, FEATURE_NAMES[i])
     for i in range(len(FEATURE_NAMES))
     if i not in (_IDX_FREQ, _IDX_FREQ_MAG)]
    + [({_IDX_FREQ, _IDX_FREQ_MAG}, "dom_freq+freq_mag")]
)
N_ABLATE = len(ABLATE_SETS)

# Progress total = ablation variants + post-process steps + final combined
N_PP     = len(PP_STEPS)
N_TOTAL  = N_ABLATE + N_PP + 1   # +1 for final_combined

# ── Global state ─────────────────────────────────────────────────────────────

_lock  = threading.Lock()
_state: dict = {
    "pipeline":       None,
    "source_joints":  None,
    "M_output":       None,
    "out_dir":        None,
    "converter":      None,
    "busy":           False,
    "progress":       {"current": 0, "total": N_TOTAL, "label": ""},
}

app = Flask(__name__)

# ── HTML ─────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ablation Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0a0a12;color:#e8e6ff;font-family:'Segoe UI',Arial,sans-serif;padding:28px}
  h1{font-size:18px;font-weight:300;letter-spacing:3px;color:#c8c6ff;margin-bottom:12px}

  /* progress */
  #progress-wrap{margin-bottom:16px;display:none}
  #progress-label{font-size:11px;color:#8b8aa8;letter-spacing:1px;margin-bottom:5px}
  #progress-track{width:100%;height:5px;background:#1e1e30;border-radius:3px;overflow:hidden}
  #progress-fill{height:100%;width:0%;background:linear-gradient(90deg,#3a20c0,#e8b880);border-radius:3px;transition:width .3s ease}

  #status{font-size:12px;color:#8b8aa8;margin-bottom:20px;min-height:16px}

  /* section labels */
  .section-label{
    font-size:10px;letter-spacing:2.5px;text-transform:uppercase;
    color:#5a5a7a;margin-bottom:10px;margin-top:24px;padding-bottom:4px;
    border-bottom:1px solid #1e1e30
  }

  /* reference row — 3 equal large cols */
  #ref-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}

  /* post-process row — auto-fill medium cols */
  #pp-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}

  /* ablation grid */
  #abl-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;margin-bottom:28px}

  /* shared card */
  .card{
    background:#13131f;border:1px solid #2a2a3f;border-radius:10px;
    padding:14px;display:flex;flex-direction:column;align-items:center;gap:10px
  }
  .card.ref{border-color:#3a3a5f}
  .card video{width:100%;border-radius:6px;background:#0f0f18}
  .card-title{font-size:11px;color:#8b8aa8;letter-spacing:1.5px;text-transform:uppercase;align-self:flex-start}
  .card.ref .card-title{color:#c0bfdf;font-size:12px}

  /* alpha controls */
  .controls{display:flex;align-items:center;gap:12px}
  .btn-alpha{
    width:32px;height:32px;border:1px solid #3a3a5f;background:#1e1e30;
    color:#e8e6ff;font-size:20px;border-radius:6px;cursor:pointer;line-height:1;
    transition:background .15s
  }
  .btn-alpha:hover{background:#2a2a45}
  .alpha-val{font-size:20px;font-weight:300;min-width:40px;text-align:center;color:#e8b880}

  /* footer */
  #footer{display:flex;justify-content:center;gap:16px;padding-top:8px}
  .footer-btn{
    padding:13px 44px;border-radius:8px;font-size:13px;letter-spacing:2px;
    text-transform:uppercase;cursor:pointer;transition:background .2s
  }
  #run-btn{background:#2e2060;color:#e8e6ff;border:1px solid #5a40c0}
  #run-btn:hover{background:#3a2878}
  #run-btn:disabled{opacity:.4;cursor:not-allowed}
  #save-btn{background:#1a2e1a;color:#a0e0a0;border:1px solid #3a6a3a}
  #save-btn:hover{background:#223a22}
</style>
</head>
<body>
<h1>ABLATION DASHBOARD</h1>

<div id="progress-wrap">
  <div id="progress-label">Initialising&hellip;</div>
  <div id="progress-track"><div id="progress-fill"></div></div>
</div>
<div id="status">Loading&hellip;</div>

<!-- ── Row 1: reference ── -->
<div class="section-label">Reference</div>
<div id="ref-grid">
  <div class="card ref">
    <div class="card-title">Source</div>
    <video id="vid-source" src="/video/source?t=0" autoplay loop muted playsinline></video>
  </div>
  <div class="card ref">
    <div class="card-title">Final Combined</div>
    <video id="vid-final_combined" src="/video/final_combined?t=0" autoplay loop muted playsinline></video>
  </div>
  <div class="card ref">
    <div class="card-title">Original Blended</div>
    <video id="vid-original_blended" src="/video/original_blended?t=0" autoplay loop muted playsinline></video>
  </div>
</div>

<!-- ── Row 2: post-processing steps ── -->
<div class="section-label">Post-processing steps</div>
<div id="pp-grid">__PP_CARDS__</div>

<!-- ── Row 3: ablation ── -->
<div class="section-label">Ablation</div>
<div id="abl-grid"></div>

<div id="footer">
  <button id="run-btn" class="footer-btn" disabled onclick="runAblation()">RUN</button>
  <button id="save-btn" class="footer-btn" onclick="saveAlphas()">SAVE</button>
</div>

<script>
const FEATURES = __FEATURES__;
const PP_KEYS  = __PP_KEYS__;
const N        = FEATURES.length;
const N_TOTAL  = N + PP_KEYS.length + 1;
const alphas   = {};
FEATURES.forEach(f => { alphas[f] = 0.5; });

function clamp(v){ return Math.round(Math.max(0,Math.min(1,v))*10)/10; }
function step(name, delta){
  alphas[name] = clamp(alphas[name]+delta);
  document.getElementById('av-'+name).textContent = alphas[name].toFixed(1);
}
function setStatus(msg){ document.getElementById('status').textContent = msg; }

function setProgress(current, total, label){
  const wrap = document.getElementById('progress-wrap');
  if(current >= total){ wrap.style.display='none'; return; }
  wrap.style.display = 'block';
  document.getElementById('progress-label').textContent =
    (label ? label+'  ' : '') + `(${current}/${total})`;
  document.getElementById('progress-fill').style.width =
    (total>0 ? current/total*100 : 0)+'%';
}

function reloadVid(id, ts){
  const v = document.getElementById('vid-'+id);
  if(v){ v.src=`/video/${encodeURIComponent(id)}?t=${ts||Date.now()}`; v.load(); v.play(); }
}

function buildAblGrid(videoTs){
  const grid = document.getElementById('abl-grid');
  grid.innerHTML = '';
  FEATURES.forEach(name => {
    const ts   = videoTs[name]||0;
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <div class="card-title">${name}</div>
      <video id="vid-${name}" src="/video/${encodeURIComponent(name)}?t=${ts}"
             autoplay loop muted playsinline></video>
      <div class="controls">
        <button class="btn-alpha" onclick="step('${name}',-0.1)">&#8722;</button>
        <span class="alpha-val" id="av-${name}">${alphas[name].toFixed(1)}</span>
        <button class="btn-alpha" onclick="step('${name}',+0.1)">&#43;</button>
      </div>`;
    grid.appendChild(card);
  });
}

function refreshAll(videoTs){
  ['source','final_combined','original_blended'].forEach(id => reloadVid(id, videoTs[id]));
  PP_KEYS.forEach(k => reloadVid('pp_'+k, videoTs['pp_'+k]));
  FEATURES.forEach(name => reloadVid(name, videoTs[name]));
}

let _pollTimer = null;
function startPolling(){
  _pollTimer = setInterval(async()=>{
    try{
      const r = await fetch('/progress');
      const d = await r.json();
      setProgress(d.current, d.total, d.label);
      if(d.current >= d.total) stopPolling();
    }catch(_){}
  }, 300);
}
function stopPolling(){ if(_pollTimer){clearInterval(_pollTimer);_pollTimer=null;} setProgress(N_TOTAL,N_TOTAL,''); }

async function postRun(){
  const r = await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({alphas})});
  return r.json();
}

async function runAblation(){
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  setStatus('Running…');
  setProgress(0,N_TOTAL,'starting…');
  startPolling();
  try{
    const d = await postRun();
    stopPolling();
    if(d.error){ setStatus('Error: '+d.error); }
    else{ refreshAll(d.videos); setStatus('Done in '+d.elapsed.toFixed(1)+'s'); }
  }catch(e){ stopPolling(); setStatus('Error: '+e); }
  btn.disabled = false;
}

async function saveAlphas(){
  try{
    const r = await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({alphas})});
    const d = await r.json();
    setStatus(d.error ? 'Save error: '+d.error : 'Saved → '+d.path);
  }catch(e){ setStatus('Save error: '+e); }
}

// Initial auto-run
(async()=>{
  setProgress(0,N_TOTAL,'initialising…');
  startPolling();
  try{
    const d = await postRun();
    stopPolling();
    if(d.error){ setStatus('Error: '+d.error); }
    else{
      buildAblGrid(d.videos);
      refreshAll(d.videos);
      setStatus('Ready — adjust alphas and click RUN.');
      document.getElementById('run-btn').disabled = false;
    }
  }catch(e){ stopPolling(); setStatus('Error: '+e); }
})();
</script>
</body>
</html>
"""


def _build_html() -> str:
    ablate_names = [name for _, name in ABLATE_SETS]
    pp_keys      = STEP_KEYS

    # Build static post-process cards (no alpha controls — steps are always applied)
    pp_cards = ""
    for key, display, _ in PP_STEPS:
        pp_cards += f"""
  <div class="card">
    <div class="card-title">{display}</div>
    <video id="vid-pp_{key}" src="/video/pp_{key}?t=0" autoplay loop muted playsinline></video>
  </div>"""

    return (
        _HTML
        .replace("__FEATURES__",  json.dumps(ablate_names))
        .replace("__PP_KEYS__",   json.dumps(pp_keys))
        .replace("__PP_CARDS__",  pp_cards)
    )


# ── Flask routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return _build_html(), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/video/<name>")
def serve_video(name):
    path = Path(_state["out_dir"]) / f"{name}.mp4"
    if not path.exists():
        return "", 404
    return send_file(str(path.resolve()), mimetype="video/mp4")


@app.route("/progress")
def progress_endpoint():
    p = _state["progress"]
    return jsonify(p)


@app.route("/run", methods=["POST"])
def run_endpoint():
    with _lock:
        if _state["busy"]:
            return jsonify({"error": "pipeline busy — try again shortly"})
        if _state["source_joints"] is None:
            return jsonify({"error": "pipeline not ready yet"})
        _state["busy"] = True

    _state["progress"] = {"current": 0, "total": N_TOTAL, "label": "starting…"}
    body   = request.get_json(force=True) or {}
    alphas = body.get("alphas", {})
    t0     = time.time()

    try:
        video_ts = _run_pipeline(alphas)
    except Exception as exc:
        with _lock:
            _state["busy"] = False
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)})

    with _lock:
        _state["busy"] = False
    _state["progress"] = {"current": N_TOTAL, "total": N_TOTAL, "label": "done"}
    return jsonify({"videos": video_ts, "elapsed": time.time() - t0})


@app.route("/save", methods=["POST"])
def save_endpoint():
    body   = request.get_json(force=True) or {}
    alphas = body.get("alphas", {})
    path   = Path(_state["out_dir"]) / "alphas.json"
    try:
        path.write_text(json.dumps(alphas, indent=2))
        return jsonify({"path": str(path)})
    except Exception as exc:
        return jsonify({"error": str(exc)})


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _render_video(joints: np.ndarray, name: str, title: str) -> None:
    out_dir   = Path(_state["out_dir"])
    converter = _state["converter"]
    _, sj     = converter.convert(joints, filename=str(out_dir / f"{name}.bvh"),
                                  iterations=100, foot_ik=False)
    plot_3d_motion(str(out_dir / f"{name}.mp4"), t2m_kinematic_chain, sj,
                   title=title, fps=20)


def _run_pipeline(alphas: dict) -> dict[str, int]:
    """
    1. Reconstruct each ablation variant (autoregressive, per-feature).
    2. Combine variants into original_blended (averaged delta).
    3. Apply each post-processing step individually to original_blended.
    4. Apply the full post-processing pipeline to get final_combined.

    Returns {video_name: unix_timestamp} for cache-busting.
    """
    pipeline      = _state["pipeline"]
    source_joints = _state["source_joints"]
    M_output      = _state["M_output"]
    ts: dict[str, int] = {}
    step_n = [0]

    def tick(label: str) -> None:
        step_n[0] += 1
        _state["progress"] = {"current": step_n[0], "total": N_TOTAL, "label": label}

    # ── Step A: ablation variants ──────────────────────────────────────────
    delta_M = np.zeros_like(source_joints, dtype=np.float32)

    for active_set, name in ABLATE_SETS:
        alpha   = float(alphas.get(name, 0.5))
        ablated = pipeline.blender.reconstruct(
            M_output, source_joints, active_features=active_set
        )
        output  = (1.0 - alpha) * source_joints + alpha * ablated
        delta_M += alpha * (ablated - source_joints)

        _render_video(output, name, f"{name}  α={alpha:.1f}")
        ts[name] = int(time.time())
        print(f"  [{name}]  α={alpha:.1f}")
        tick(name)

    # ── Step B: original blended (averaged delta, no post-processing) ──────
    original_blended = source_joints + delta_M / N_ABLATE
    _render_video(original_blended, "original_blended", "Original blended")
    ts["original_blended"] = int(time.time())
    print("  [original_blended]")

    # ── Step C: individual post-processing steps ───────────────────────────
    for key, display, _ in PP_STEPS:
        processed = apply_pipeline(original_blended, reference=source_joints,
                                   enabled=[key])
        _render_video(processed, f"pp_{key}", display)
        ts[f"pp_{key}"] = int(time.time())
        print(f"  [pp_{key}]  {display}")
        tick(display)

    # ── Step D: final combined (full pipeline) ─────────────────────────────
    final = apply_pipeline(original_blended, reference=source_joints, enabled=None)
    _render_video(final, "final_combined", "Final combined")
    ts["final_combined"] = int(time.time())
    print("  [final_combined]")
    tick("final combined")

    return ts


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_pipeline_models(args) -> ZoneBlendPipeline:
    device   = torch.device("cpu" if args.gpu_id == -1 else f"cuda:{args.gpu_id}")
    dim_pose = 251 if args.dataset_name == "kit" else 263

    model_opt = get_opt(
        os.path.join(args.checkpoints_dir, args.dataset_name, args.name, "opt.txt"),
        device=device,
    )
    vq_opt_path      = os.path.join(args.checkpoints_dir, args.dataset_name,
                                    model_opt.vq_name, "opt.txt")
    vq_opt           = get_opt(vq_opt_path, device=device)
    vq_opt.dim_pose  = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens     = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim       = vq_opt.code_dim

    class _Opt: pass
    opt = _Opt()
    opt.device = device; opt.gpu_id = args.gpu_id
    opt.name = args.name; opt.res_name = args.res_name
    opt.dataset_name = args.dataset_name
    opt.checkpoints_dir = args.checkpoints_dir
    opt.time_steps = args.time_steps; opt.cond_scale = args.cond_scale
    opt.temperature = args.temperature; opt.topkr = args.topkr
    opt.gumbel_sample = False; opt.force_mask = False

    res_opt   = get_opt(
        os.path.join(args.checkpoints_dir, args.dataset_name, args.res_name, "opt.txt"),
        device=device,
    )
    res_model       = load_res_model(res_opt, vq_opt, opt)
    t2m_transformer = load_trans_model(model_opt, opt, "latest.tar")

    mean = np.load(os.path.join(args.checkpoints_dir, args.dataset_name,
                                model_opt.vq_name, "meta", "mean.npy"))
    std  = np.load(os.path.join(args.checkpoints_dir, args.dataset_name,
                                model_opt.vq_name, "meta", "std.npy"))

    for m in (vq_model, t2m_transformer, res_model):
        m.eval(); m.to(device)

    return ZoneBlendPipeline(
        vq_model=vq_model, mask_transformer=t2m_transformer, res_model=res_model,
        vq_opt=vq_opt, mean=mean, std=std,
        alpha=args.alpha, zone_mode=args.zone_mode, device=device,
        time_steps=args.time_steps, cond_scale=args.cond_scale,
        temperature=args.temperature, topkr=args.topkr,
    )


def _load_source(path: str):
    data       = np.load(path, allow_pickle=True)
    motion_vec = data["motion"].astype(np.float32) if "motion" in data else None
    if "joints" in data:
        joints = data["joints"].astype(np.float32)
    elif motion_vec is not None:
        joints = recover_from_ric(torch.from_numpy(motion_vec).float(), 22).numpy()
    else:
        raise ValueError(f"{path} must contain 'motion' or 'joints'.")
    return motion_vec, joints


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs_file",       required=True)
    p.add_argument("--job_idx",         type=int,   default=0)
    p.add_argument("--alpha",           type=float, default=0.5)
    p.add_argument("--zone_mode",       default="standard",
                   choices=["standard", "side_specific"])
    p.add_argument("--gpu_id",          type=int,   default=0)
    p.add_argument("--name",            default="MaskTransformer")
    p.add_argument("--res_name",        default="ResTransformer")
    p.add_argument("--dataset_name",    default="t2m")
    p.add_argument("--checkpoints_dir", default="./checkpoints")
    p.add_argument("--time_steps",      type=int,   default=18)
    p.add_argument("--cond_scale",      type=float, default=4.0)
    p.add_argument("--temperature",     type=float, default=1.0)
    p.add_argument("--topkr",           type=float, default=0.9)
    p.add_argument("--out_dir",         default="./dashboard_outputs")
    p.add_argument("--port",            type=int,   default=5050)
    args = p.parse_args()

    fixseed(10107)

    with open(args.jobs_file, encoding="utf-8-sig") as f:
        jobs = json.load(f)
    job = jobs[args.job_idx]
    print(f"\nJob [{args.job_idx}]: {job['ext']}")
    print(f"  prompt : {job['text_prompt']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _state["out_dir"]   = str(out_dir)
    _state["converter"] = Joint2BVHConvertor()

    print("\nLoading models...")
    pipeline = _load_pipeline_models(args)
    _state["pipeline"] = pipeline

    src_path = job.get("source_motion")
    if not src_path:
        from scripts.run_zone_blend import run_video_bridge
        src_path = run_video_bridge(job["source_video"])
    motion_vec, orig_joints = _load_source(src_path)

    print("\nRunning MoMask inference (once)...")
    if motion_vec is not None:
        src_j, _, _, M_output = pipeline.run_from_motion_vec(
            motion_vec, orig_joints, job["text_prompt"], return_intermediates=True
        )
    else:
        src_j, _, _, M_output = pipeline.run(
            orig_joints, job["text_prompt"], return_intermediates=True
        )

    _state["source_joints"] = src_j
    _state["M_output"]      = M_output

    print("Rendering source reference video...")
    _render_video(src_j, "source", "Source")
    print("Done.\n")

    url = f"http://localhost:{args.port}"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Dashboard → {url}  (Ctrl+C to quit)\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
