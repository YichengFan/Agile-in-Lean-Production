# app.py
# Streamlit control panel for LEGO Lean DES (Pull mode: CONWIP + Kanban) with KPI charts.

import copy
import json
import time
from typing import Dict, Any

import streamlit as st
import pandas as pd

from env import LegoLeanEnv, CONFIG as DEFAULT_CONFIG


# ---------------------------
# Helpers
# ---------------------------

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
# App UI
# ---------------------------

st.set_page_config(page_title="LEGO Lean Simulator", layout="wide")
st.title("LEGO Lean Production Simulator (Push/Pull: CONWIP + Kanban)")

cfg = copy.deepcopy(DEFAULT_CONFIG)

with st.sidebar:
    st.header("Run Controls")
    enable_pull = st.checkbox("Enable Pull (CONWIP + Kanban)", value=True)
    if enable_pull:
        st.markdown("### Pull Control (CONWIP + Kanban)")
        conwip_cap = st.number_input("CONWIP WIP cap", min_value=0,
                                     value=int(cfg["parameters"].get("conwip_wip_cap", 12)), step=1)
        auto_release = st.checkbox("Closed-loop CONWIP (auto release on finish)",
                                   value=bool(cfg["parameters"].get("auto_release_conwip", True)))

        # release stages
        release_stage_ids = st.multiselect(
            "Release stage(s) (token-gated)",
            options=[s["stage_id"] for s in cfg["stages"]],
            default=cfg["parameters"].get("release_stage_ids", ["S1"])
        )

        st.markdown("**Kanban caps (control limits)**")
        kc1 = st.number_input("Kanban cap: C3", min_value=0,
                              value=int(cfg["parameters"].get("kanban_caps", {}).get("C3", 4)), step=1)
        kd1 = st.number_input("Kanban cap: D1", min_value=0,
                              value=int(cfg["parameters"].get("kanban_caps", {}).get("D1", 4)), step=1)
        kd2 = st.number_input("Kanban cap: D2", min_value=0,
                              value=int(cfg["parameters"].get("kanban_caps", {}).get("D2", 4)), step=1)
    sim_time = st.number_input("Simulation time (seconds)", min_value=1, value=3600, step=60)

    use_random_seed = st.checkbox("Use random seed", value=False)
    seed = None
    if not use_random_seed:
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    # Pull-mode release: initial injection only (then closed-loop CONWIP can replenish)
    initial_release = st.number_input("Initial release qty (CONWIP)", min_value=0, value=12, step=1)

    deterministic = st.checkbox("Deterministic processing (override times; disable disruptions)", value=False)
    show_logs_n = st.number_input("Show last N logs", min_value=0, value=30, step=5)


st.subheader("Global Parameters")
c1, c2 = st.columns(2)
with c1:
    takt = st.number_input(
        "Target takt (sec)",
        min_value=0.0,
        value=float(cfg["parameters"].get("target_takt_sec", 10.0)),
        step=0.5
    )
    cfg["parameters"]["target_takt_sec"] = float(takt)
with c2:
    sample_dt = st.number_input(
        "Timeline sample Δt (sec)",
        min_value=0.5,
        value=float(cfg["parameters"].get("timeline_sample_dt_sec", 5.0)),
        step=0.5
    )
    cfg["parameters"]["timeline_sample_dt_sec"] = float(sample_dt)


st.markdown("---")
st.subheader("Pull Control (CONWIP + Kanban)")

p1, p2, p3 = st.columns(3)
with p1:
    conwip_cap = st.number_input(
        "CONWIP WIP cap",
        min_value=0,
        value=int(cfg["parameters"].get("conwip_wip_cap", 12) or 0),
        step=1
    )
    # If cap==0 -> treat as disabled by setting None (optional convention)
    cfg["parameters"]["conwip_wip_cap"] = None if int(conwip_cap) <= 0 else int(conwip_cap)

with p2:
    auto_release = st.checkbox(
        "Closed-loop CONWIP (auto release on finish)",
        value=bool(cfg["parameters"].get("auto_release_conwip", True))
    )
    cfg["parameters"]["auto_release_conwip"] = bool(auto_release)

with p3:
    # release_stage_ids: allow selecting which stages are gated by order tokens
    all_stage_ids = [s["stage_id"] for s in cfg["stages"]]
    default_release = cfg["parameters"].get("release_stage_ids", ["S1"])
    release_stage_ids = st.multiselect(
        "Release stage(s) (token-gated)",
        options=all_stage_ids,
        default=default_release if default_release else ["S1"]
    )
    cfg["parameters"]["release_stage_ids"] = list(release_stage_ids)


st.markdown("**Kanban caps (control limits, not physical capacities)**")
kcfg = cfg["parameters"].get("kanban_caps", {}) or {}
k1, k2, k3, k4 = st.columns(4)

with k1:
    cap_c3 = st.number_input("Kanban cap: C3", min_value=0, value=int(kcfg.get("C3", 4)), step=1)
with k2:
    cap_d1 = st.number_input("Kanban cap: D1", min_value=0, value=int(kcfg.get("D1", 4)), step=1)
with k3:
    cap_d2 = st.number_input("Kanban cap: D2", min_value=0, value=int(kcfg.get("D2", 4)), step=1)
with k4:
    # Optional: allow user to paste full dict for advanced control
    adv_kanban = st.text_area(
        "Kanban caps override (JSON dict, optional)",
        value="",
        height=80,
        help='Example: {"C1": 5, "C2": 5, "C3": 4, "D1": 4, "D2": 4}'
    )

kanban_caps = {"C3": int(cap_c3), "D1": int(cap_d1), "D2": int(cap_d2)}
if adv_kanban.strip():
    try:
        parsed = json.loads(adv_kanban)
        if isinstance(parsed, dict):
            # Merge/override
            for k, v in parsed.items():
                kanban_caps[str(k)] = int(v)
    except Exception:
        st.warning("Kanban caps override JSON invalid; ignored.")

cfg["parameters"]["kanban_caps"] = kanban_caps


st.markdown("---")
st.subheader("Shift Window")
d1, d2 = st.columns(2)
with d1:
    shift_start_min = st.number_input(
        "Shift start (minutes in day)",
        min_value=0,
        max_value=24 * 60,
        value=int(cfg["shift_schedule"][0]["start_minute"])
    )
with d2:
    shift_end_min = st.number_input(
        "Shift end (minutes in day)",
        min_value=0,
        max_value=24 * 60,
        value=int(cfg["shift_schedule"][0]["end_minute"])
    )
cfg["shift_schedule"][0]["start_minute"] = int(shift_start_min)
cfg["shift_schedule"][0]["end_minute"] = int(shift_end_min)


st.markdown("---")
st.subheader("Processing & Quality")
exp = st.expander("Per-stage timing, transport, defects", expanded=False)

def stage_controls(label: str, sid: str):
    st.markdown(f"**{label} ({sid})**")
    s = stage_by_id(cfg, sid)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        base = st.number_input(
            f"{sid} base_time/worker (sec)",
            min_value=0.0,
            value=float(s.get("base_process_time_sec", 1.0)),
            step=0.5,
            key=f"{sid}_base"
        )
        s["base_process_time_sec"] = float(base)
    with c2:
        workers = st.number_input(
            f"{sid} workers",
            min_value=1,
            value=int(s.get("workers", 1)),
            step=1,
            key=f"{sid}_workers"
        )
        s["workers"] = int(workers)
    with c3:
        dist = st.selectbox(
            f"{sid} dist.type",
            options=["constant", "triangular", "normal", "uniform", "lognormal", "exponential"],
            index=0,
            key=f"{sid}_dist"
        )
        if "time_distribution" not in s or s.get("time_distribution") is None:
            s["time_distribution"] = {}
        s["time_distribution"]["type"] = dist
    with c4:
        p1 = st.number_input(
            f"{sid} p1",
            value=float(s["time_distribution"].get("p1") or (base * 0.5 if dist == "triangular" else base)),
            step=0.5,
            key=f"{sid}_p1"
        )
        s["time_distribution"]["p1"] = float(p1)
    with c5:
        p2 = st.number_input(
            f"{sid} p2",
            value=float(s["time_distribution"].get("p2") or (base if dist == "triangular" else base * 0.1)),
            step=0.5,
            key=f"{sid}_p2"
        )
        s["time_distribution"]["p2"] = float(p2)

    e1, e2, e3 = st.columns(3)
    with e1:
        p3 = st.number_input(
            f"{sid} p3",
            value=float(s["time_distribution"].get("p3") or (base * 1.5 if dist == "triangular" else 0.0)),
            step=0.5,
            key=f"{sid}_p3"
        )
        s["time_distribution"]["p3"] = float(p3)
    with e2:
        trans = st.number_input(
            f"{sid} transport (sec)",
            min_value=0.0,
            value=float(s.get("transport_time_sec", 0.0)),
            step=0.1,
            key=f"{sid}_tt"
        )
        s["transport_time_sec"] = float(trans)
    with e3:
        defect = st.number_input(
            f"{sid} defect rate",
            min_value=0.0,
            max_value=1.0,
            value=float(s.get("defect_rate", 0.0)),
            step=0.01,
            key=f"{sid}_def"
        )
        s["defect_rate"] = float(defect)

with exp:
    stage_controls("Type Sorting", "S1")
    stage_controls("Set Sorting", "S2")

    # --- S2 per-output transport times ---
    st.markdown("**S2 per-output transport (sec)**")
    s2 = stage_by_id(cfg, "S2")
    if "transport_time_to_outputs_sec" not in s2 or s2.get("transport_time_to_outputs_sec") is None:
        base = float(s2.get("transport_time_sec", 0.0))
        s2["transport_time_to_outputs_sec"] = {"C1": base, "C2": base, "C3": base}

    ttc1, ttc2, ttc3 = st.columns(3)
    with ttc1:
        s2_c1 = st.number_input("S2 → C1", min_value=0.0, value=float(s2["transport_time_to_outputs_sec"].get("C1", 0.0)), step=0.1, key="S2_to_C1")
    with ttc2:
        s2_c2 = st.number_input("S2 → C2", min_value=0.0, value=float(s2["transport_time_to_outputs_sec"].get("C2", 0.0)), step=0.1, key="S2_to_C2")
    with ttc3:
        s2_c3 = st.number_input("S2 → C3", min_value=0.0, value=float(s2["transport_time_to_outputs_sec"].get("C3", 0.0)), step=0.1, key="S2_to_C3")

    s2["transport_time_to_outputs_sec"]["C1"] = float(s2_c1)
    s2["transport_time_to_outputs_sec"]["C2"] = float(s2_c2)
    s2["transport_time_to_outputs_sec"]["C3"] = float(s2_c3)

    stage_controls("Axis Assembly", "S3")
    stage_controls("Chassis Assembly", "S4")
    stage_controls("Final Assembly", "S5")


st.markdown("---")
st.subheader("Random Disruptions")
c1, c2 = st.columns(2)
with c1:
    miss_p = st.number_input(
        "missing_brick_prob",
        min_value=0.0,
        max_value=1.0,
        value=float(cfg["random_events"].get("missing_brick_prob", 0.1)),
        step=0.01
    )
    cfg["random_events"]["missing_brick_prob"] = float(miss_p)
with c2:
    miss_pen = st.number_input(
        "missing_brick_penalty_sec",
        min_value=0.0,
        value=float(cfg["random_events"].get("missing_brick_penalty_sec", 2.0)),
        step=0.5
    )
    cfg["random_events"]["missing_brick_penalty_sec"] = float(miss_pen)


st.markdown("---")
st.subheader("Initial Buffer Stocks")
buffer_ids = ["A", "B", "C1", "C2", "C3", "D1", "D2", "E"]
bcols = st.columns(len(buffer_ids))
for i, bid in enumerate(buffer_ids):
    with bcols[i]:
        b = buffer_by_id(cfg, bid)
        default_str = json.dumps(b.get("initial_stock", {}), indent=0)
        txt = st.text_area(f"{bid} initial_stock (json dict)", value=default_str, height=120)
        try:
            parsed = json.loads(txt)
            if isinstance(parsed, dict):
                b["initial_stock"] = {str(k): int(v) for k, v in parsed.items()}
            else:
                st.warning(f"{bid} initial_stock must be a JSON object/dict. Ignored.")
        except Exception:
            st.warning(f"{bid} initial_stock is invalid JSON. Ignored.")


# ---------------------------
# Run
# ---------------------------

if st.button("Run Simulation"):
    t0 = time.time()

    # Apply deterministic override if selected
    if deterministic:
        for s in cfg["stages"]:
            s["time_distribution"] = {"type": "constant"}
        cfg["random_events"]["missing_brick_prob"] = 0.0

    # Assembly trace toggle
    trace_assembly = st.sidebar.checkbox("Trace assembly (store per-job consumed/produced)", value=False)
    cfg["parameters"]["trace_assembly"] = bool(trace_assembly)
    # Apply mode settings
    if enable_pull:
        cfg["parameters"]["release_stage_ids"] = release_stage_ids
        cfg["parameters"]["conwip_wip_cap"] = int(conwip_cap) if int(conwip_cap) > 0 else None
        cfg["parameters"]["auto_release_conwip"] = bool(auto_release)
        cfg["parameters"]["kanban_caps"] = {"C3": int(kc1), "D1": int(kd1), "D2": int(kd2)}
    else:
        # Disable pull control explicitly (back to push)
        cfg["parameters"]["release_stage_ids"] = []  # or ["S1"] but with no gating
        cfg["parameters"]["conwip_wip_cap"] = None
        cfg["parameters"]["auto_release_conwip"] = False
        cfg["parameters"]["kanban_caps"] = {}

    # Build env
    env = LegoLeanEnv(cfg, time_unit="sec", seed=(None if seed is None else int(seed)))

    # Pull-mode: initial release only (then closed-loop CONWIP replenishes if enabled)
    if int(initial_release) > 0:
        env.enqueue_orders(qty=int(initial_release))

    #2026-01-01 存疑，如果是push的话enqueue激活就够了，这样是否会反复激活S1？
    # Kick off: try starting all stages once
    for s in env.stages.values():
        env._push_event(env.t, "try_start", {"stage_id": s.stage_id})

    env.run_for(float(sim_time), max_events=1_000_000)
    t1 = time.time()

    st.success(f"Simulation finished in {t1 - t0:.3f} sec (wall).")

    kpis = env.get_kpis()
    st.subheader("KPIs")
    st.json(kpis)

    # Pull-control diagnostics (if present in env)
    st.subheader("Pull Diagnostics")
    diag = {
        "conwip_wip_cap": cfg["parameters"].get("conwip_wip_cap"),
        "auto_release_conwip": cfg["parameters"].get("auto_release_conwip"),
        "release_stage_ids": cfg["parameters"].get("release_stage_ids"),
        "kanban_caps": cfg["parameters"].get("kanban_caps"),
    }
    # Optional runtime counters
    if hasattr(env, "kanban_blocking_counts"):
        diag["kanban_blocking_counts"] = getattr(env, "kanban_blocking_counts")
    st.json(diag)

    util_rows = [{"team_id": k, "utilization": round(v, 4)} for k, v in kpis.get("utilization_per_team", {}).items()]
    if util_rows:
        st.table(util_rows)

    # --- Charts ---
    st.subheader("KPI Time Series")
    if env.timeline:
        df = pd.DataFrame(env.timeline).set_index("t")
        st.markdown("**WIP and Finished**")
        st.line_chart(df[["wip", "finished"]], height=220)

        st.markdown("**Throughput (units/min)**")
        st.line_chart(df[["throughput_per_min"]], height=220)

        buffer_cols = [c for c in ["B", "C1", "C2", "C3", "D1", "D2", "E"] if c in df.columns]
        if buffer_cols:
            st.markdown("**Buffer Levels**")
            st.line_chart(df[buffer_cols], height=260)
    else:
        st.info("No timeline captured. Increase simulation time or decrease sample interval.")

    # Downloads
    st.subheader("Artifacts")
    st.download_button(
        "Download used config (JSON)",
        data=json.dumps(cfg, indent=2),
        file_name="used_config.json",
        mime="application/json"
    )
    st.download_button(
        "Download logs (txt)",
        data="\n".join(env.log),
        file_name="trace_log.txt",
        mime="text/plain"
    )
    if cfg["parameters"].get("trace_assembly"):
        st.download_button(
            "Download assembly traces (json)",
            data=json.dumps(env.assembly_traces, indent=2),
            file_name="assembly_traces.json",
            mime="application/json"
        )
else:
    st.info("Adjust parameters, then click **Run Simulation**.")
