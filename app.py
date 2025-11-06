# app.py
# Streamlit control panel for the LEGO Lean DES environment with CONWIP (Pull).
# - Lets you edit parameters (CONWIP cap, routing, processing times, disruptions, shifts)
# - Runs the simulation and displays KPIs and logs
# - No external files required beyond env.py

import copy
import json
import time
from typing import Dict, Any, List

import streamlit as st

# Import your environment and default configuration
from env import LegoLeanEnv, CONFIG as DEFAULT_CONFIG

# ---------------------------
# Helpers
# ---------------------------

def normalize_probs(values: List[float]) -> List[float]:
    total = sum(max(0.0, v) for v in values)
    if total <= 0:
        # fallback to equal split if all zero/negative
        n = len(values)
        return [1.0 / n] * n
    return [max(0.0, v) / total for v in values]

def stage_by_id(cfg: Dict[str, Any], sid: str) -> Dict[str, Any]:
    for s in cfg["stages"]:
        if s["stage_id"] == sid:
            return s
    raise KeyError(f"Stage not found: {sid}")

def buffer_by_id(cfg: Dict[str, Any], bid: str) -> Dict[str, Any]:
    for b in cfg["buffers"]:
        if b["buffer_id"] == bid:
            return b
    raise KeyError(f"Buffer not found: {bid}")

# ---------------------------
# App layout
# ---------------------------

st.set_page_config(page_title="LEGO Lean (CONWIP) Simulator", layout="wide")
st.title("LEGO Lean Production Simulator (CONWIP Pull)")

with st.sidebar:
    st.header("Run Controls")
    sim_time = st.number_input("Simulation time (seconds)", min_value=1, value=3600, step=60)
    seed = st.number_input("Random seed (None = random)", min_value=0, value=42, step=1)
    show_logs_n = st.number_input("Show last N logs", min_value=0, value=30, step=5)

# Make a working copy of the default config
cfg = copy.deepcopy(DEFAULT_CONFIG)

st.subheader("Global Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    conwip_cap = st.number_input("CONWIP cap K (None disables)", min_value=0, value=int(cfg["parameters"].get("conwip_cap", 50)))
    cfg["parameters"]["conwip_cap"] = int(conwip_cap)
with col2:
    takt = st.number_input("Target takt (sec)", min_value=0.0, value=float(cfg["parameters"].get("target_takt_sec", 10.0)), step=0.5)
    cfg["parameters"]["target_takt_sec"] = takt
with col3:
    source_ids_default = cfg["parameters"].get("source_stage_ids", [])
    all_sources = [s["stage_id"] for s in cfg["stages"] if not s.get("input_buffers")]
    if not source_ids_default:
        source_ids_default = all_sources
    source_stage_ids = st.multiselect("Source stages (for CONWIP release)", options=all_sources, default=source_ids_default)
    cfg["parameters"]["source_stage_ids"] = source_stage_ids

st.markdown("---")
st.subheader("Shift Window")
c1, c2 = st.columns(2)
with c1:
    shift_start_min = st.number_input("Shift start (minutes in day)", min_value=0, max_value=24*60, value=int(cfg["shift_schedule"][0]["start_minute"]))
with c2:
    shift_end_min = st.number_input("Shift end (minutes in day)", min_value=0, max_value=24*60, value=int(cfg["shift_schedule"][0]["end_minute"]))
cfg["shift_schedule"][0]["start_minute"] = int(shift_start_min)
cfg["shift_schedule"][0]["end_minute"] = int(shift_end_min)

st.markdown("---")
st.subheader("Routing (Stage S2 → C1/C2/C3)")
# Read current routing (if any)
s2 = stage_by_id(cfg, "S2")
p_c1, p_c2, p_c3 = 0.40, 0.40, 0.20
if s2.get("output_rules"):
    rules = s2["output_rules"]
    # assume order [C1, C2, C3]
    try:
        p_c1 = float([r for r in rules if r["buffer_id"] == "C1"][0]["p"])
        p_c2 = float([r for r in rules if r["buffer_id"] == "C2"][0]["p"])
        p_c3 = float([r for r in rules if r["buffer_id"] == "C3"][0]["p"])
    except Exception:
        pass

rc1, rc2, rc3 = st.columns(3)
with rc1:
    p_c1 = st.number_input("P(C1)", min_value=0.0, max_value=1.0, value=p_c1, step=0.05)
with rc2:
    p_c2 = st.number_input("P(C2)", min_value=0.0, max_value=1.0, value=p_c2, step=0.05)
with rc3:
    p_c3 = st.number_input("P(C3)", min_value=0.0, max_value=1.0, value=p_c3, step=0.05)

p_c1, p_c2, p_c3 = normalize_probs([p_c1, p_c2, p_c3])
s2["output_rules"] = [
    {"buffer_id": "C1", "p": p_c1},
    {"buffer_id": "C2", "p": p_c2},
    {"buffer_id": "C3", "p": p_c3},
]

st.caption(f"Normalized routing: C1={p_c1:.2f}, C2={p_c2:.2f}, C3={p_c3:.2f}")

st.markdown("---")
st.subheader("Processing Times & Distributions (per Stage)")
exp = st.expander("Advanced: per-stage timing, transport, defect, rework", expanded=False)

def stage_controls(label: str, sid: str):
    st.markdown(f"**{label} ({sid})**")
    s = stage_by_id(cfg, sid)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        base = st.number_input(f"{sid} base_process_time_sec", min_value=0.0, value=float(s.get("base_process_time_sec", 1.0)), step=0.5, key=f"{sid}_base")
        s["base_process_time_sec"] = base
    with c2:
        dist = st.selectbox(f"{sid} dist.type", options=["constant", "triangular", "normal", "uniform", "lognormal", "exponential"], index=0, key=f"{sid}_dist")
        s["time_distribution"]["type"] = dist
    with c3:
        p1 = st.number_input(f"{sid} dist.p1", value=float(s["time_distribution"].get("p1") or (base*0.5 if dist=="triangular" else base)), step=0.5, key=f"{sid}_p1")
        s["time_distribution"]["p1"] = p1
    with c4:
        p2 = st.number_input(f"{sid} dist.p2", value=float(s["time_distribution"].get("p2") or (base if dist=="triangular" else base*0.1)), step=0.5, key=f"{sid}_p2")
        s["time_distribution"]["p2"] = p2

    c5, c6, c7 = st.columns(3)
    with c5:
        p3 = st.number_input(f"{sid} dist.p3", value=float(s["time_distribution"].get("p3") or (base*1.5 if dist=="triangular" else 0.0)), step=0.5, key=f"{sid}_p3")
        s["time_distribution"]["p3"] = p3
    with c6:
        trans = st.number_input(f"{sid} transport_time_sec", min_value=0.0, value=float(s.get("transport_time_sec", 0.0)), step=0.1, key=f"{sid}_tt")
        s["transport_time_sec"] = trans
    with c7:
        defect = st.number_input(f"{sid} defect_rate", min_value=0.0, max_value=1.0, value=float(s.get("defect_rate", 0.0)), step=0.01, key=f"{sid}_def")
        s["defect_rate"] = defect

with exp:
    stage_controls("Type Sorting", "S1")
    stage_controls("Set Sorting", "S2")
    stage_controls("Axis Assembly", "S3")
    stage_controls("Chassis Assembly", "S4")
    stage_controls("Final Assembly", "S5")

st.markdown("---")
st.subheader("Random Disruptions")
c1, c2 = st.columns(2)
with c1:
    miss_p = st.number_input("missing_brick_prob", min_value=0.0, max_value=1.0, value=float(cfg["random_events"].get("missing_brick_prob", 0.1)), step=0.01)
    cfg["random_events"]["missing_brick_prob"] = miss_p
with c2:
    miss_pen = st.number_input("missing_brick_penalty_sec", min_value=0.0, value=float(cfg["random_events"].get("missing_brick_penalty_sec", 2.0)), step=0.5)
    cfg["random_events"]["missing_brick_penalty_sec"] = miss_pen

st.markdown("---")
st.subheader("Initial Stocks (Buffers)")
bcols = st.columns(7)
for i, bid in enumerate(["B", "C1", "C2", "C3", "D1", "D2", "E"]):
    with bcols[i]:
        b = buffer_by_id(cfg, bid)
        init_val = st.number_input(f"{bid} initial_stock", min_value=0, value=int(b.get("initial_stock", 0)))
        b["initial_stock"] = int(init_val)

# ---------------------------
# Run
# ---------------------------

run = st.button("Run Simulation")

if run:
    t0 = time.time()
    env = LegoLeanEnv(cfg, time_unit="sec", seed=int(seed))
    # CONWIP: automatically pull-to-cap at start
    env._pull_to_cap()
    # Kick-start stages trying to start
    for s in env.stages.values():
        env._push_event(env.t, "try_start", {"stage_id": s.stage_id})
    # Run
    env.run_for(float(sim_time))
    t1 = time.time()

    st.success(f"Simulation finished in {t1 - t0:.3f} sec (wall time).")

    # KPIs
    kpis = env.get_kpis()
    st.subheader("KPIs")
    st.json(kpis)

    # Utilization table
    util_items = [{"team_id": k, "utilization": v} for k, v in kpis.get("utilization_per_team", {}).items()]
    if util_items:
        st.table(util_items)

    # Logs
    st.subheader("Event Trace (tail)")
    if show_logs_n > 0:
        tail = env.log[-int(show_logs_n):]
        st.code("\n".join(tail), language="text")

    # Download config & logs
    st.subheader("Artifacts")
    st.download_button(
        label="Download used config (JSON)",
        data=json.dumps(cfg, indent=2),
        file_name="used_config.json",
        mime="application/json"
    )
    st.download_button(
        label="Download logs (txt)",
        data="\n".join(env.log),
        file_name="trace_log.txt",
        mime="text/plain"
    )
else:
    st.info("Adjust parameters on the page, then click **Run Simulation**.")
