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
        cfg["parameters"]["push_demand_enabled"] = False
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
    else:
        st.markdown("### Push Planning (Demand-based)")
        horizon_weeks = st.number_input(
            "Demand horizon (weeks)",
            min_value=1, value=int(cfg["parameters"].get("push_demand_horizon_weeks", 3)), step=1
        )
        weekly_mean = st.number_input(
            "Forecast weekly demand (units)",
            min_value=0, value=int(cfg["parameters"].get("push_weekly_demand_mean", 30)), step=1,
            help="Total weekly demand (will be allocated across models based on probability distribution)"
        )
        forecast_noise_pct = st.slider(
            "Forecast noise (±%)",
            min_value=0, max_value=50, value=int(float(cfg["parameters"].get("push_forecast_noise_pct", 0.1)) * 100),
            help="Noise applied to forecast generation"
        )
        real_noise_pct = st.slider(
            "Realization noise (±%)",
            min_value=0, max_value=50, value=int(float(cfg["parameters"].get("push_realization_noise_pct", 0.05)) * 100),
            help="Noise applied to final demand realization. Higher values create more unmet demand scenarios."
        )
        waste_pct = st.slider(
            "Procurement waste/safety stock (%)",
            min_value=0, max_value=30, value=int(float(cfg["parameters"].get("push_procurement_waste_rate", 0.05)) * 100)
        )
        # Auto-release is always enabled in push mode (forecast automatically triggers production)
        margin_per_unit = st.number_input(
            "Margin per unit",
            min_value=0.0,
            value=float(cfg["parameters"].get("cost", {}).get("margin_per_unit", 5600.0)),
            step=100.0
        )
        cfg["parameters"]["push_demand_enabled"] = True
        cfg["parameters"]["push_demand_horizon_weeks"] = int(horizon_weeks)
        cfg["parameters"]["push_weekly_demand_mean"] = float(weekly_mean)
        cfg["parameters"]["push_forecast_noise_pct"] = float(forecast_noise_pct) / 100.0
        cfg["parameters"]["push_realization_noise_pct"] = float(real_noise_pct) / 100.0
        cfg["parameters"]["push_procurement_waste_rate"] = float(waste_pct) / 100.0
        # Auto-release is always enabled in push mode (forecast automatically triggers production)
        cfg["parameters"]["push_auto_release"] = True
        cfg["parameters"]["material_cost_mode"] = "procure_forecast"
        cfg["parameters"]["release_stage_ids"] = ["S1"]
        cfg["parameters"]["conwip_wip_cap"] = None
        cfg["parameters"]["auto_release_conwip"] = False
        cfg["parameters"]["kanban_caps"] = {}
        cfg["parameters"].setdefault("cost", {})
        cfg["parameters"]["cost"]["margin_per_unit"] = float(margin_per_unit)
# Calculate simulation time based on demand horizon in push mode
    if cfg["parameters"].get("push_demand_enabled", False):
        sim_time = int(horizon_weeks) * 5 * 8 * 60  # weeks * 5 days/week * 8 hours/day * 60 min/hour
        st.metric("Simulation time (minutes)",
                 f"{sim_time:,}",
                 help=f"Automatically calculated from demand horizon: {int(horizon_weeks)} weeks × 5 days/week × 8 hours/day × 60 min/hour")
    else:
        sim_time = st.number_input("Simulation time (minutes)", min_value=1, value=7200, step=60,
                                  help="Simulation time (minutes) - adjustable in pull mode")




    use_random_seed = st.checkbox("Use random seed", value=False)
    seed = None
    #2026-1-3 原本是if not use ,改成if use
    if use_random_seed:
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    # Multi-model order release configuration
    st.markdown("### Order Release Configuration")
    
    # Model type selection with quantities
    st.markdown("**Specify quantity for each model:**")
    model_definitions = cfg.get("models", {})
    model_quantities = {}
    
    if model_definitions:
        # Create columns for better layout (2 models per row)
        model_ids = sorted(list(model_definitions.keys()))
        num_cols = 2
        num_rows = (len(model_ids) + num_cols - 1) // num_cols
        
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                model_idx = row * num_cols + col_idx
                if model_idx < len(model_ids):
                    model_id = model_ids[model_idx]
                    model_name = model_definitions[model_id].get("name", model_id)
                    with cols[col_idx]:
                        qty = st.number_input(
                            f"{model_name} ({model_id})",
                            min_value=0,
                            value=0,
                            step=1,
                            key=f"model_qty_{model_id}",
                            help=f"Quantity of {model_name} to produce"
                        )
                        if qty > 0:
                            model_quantities[model_id] = qty
        
        total_orders = sum(model_quantities.values())
        if total_orders > 0:
            st.success(f"**Total orders to release: {total_orders}**")
            # Show breakdown
            breakdown = ", ".join([f"{qty}x {model_definitions[mid].get('name', mid)}" 
                                  for mid, qty in sorted(model_quantities.items())])
            st.caption(f"Breakdown: {breakdown}")
        else:
            st.warning("⚠️ No orders specified. Please set quantity for at least one model.")
    else:
        st.info("No model definitions found in config. Using default materials.")
        # Fallback: single quantity input for backward compatibility
        initial_release = st.number_input("Initial release qty", min_value=0, value=12, step=1)
        if initial_release > 0:
            model_quantities["default"] = initial_release

    deterministic = st.checkbox("Deterministic processing (override times; disable disruptions)", value=False)
    show_logs_n = st.number_input("Show last N logs", min_value=0, value=30, step=5)


st.subheader("Global Parameters")
c1, c2 = st.columns(2)
with c1:
    takt = st.number_input(
        "Target takt (min)",
        min_value=0.0,
        value=float(cfg["parameters"].get("target_takt_min", cfg["parameters"].get("target_takt_sec", 10.0) / 60.0)),
        step=0.5
    )
    cfg["parameters"]["target_takt_min"] = float(takt)
with c2:
    sample_dt = st.number_input(
        "Timeline sample Δt (min)",
        min_value=0.5,
        value=float(cfg["parameters"].get("timeline_sample_dt_min", cfg["parameters"].get("timeline_sample_dt_sec", 5.0) / 60.0)),
        step=0.5
    )
    cfg["parameters"]["timeline_sample_dt_min"] = float(sample_dt)


if enable_pull:
    # 2026-01-09: Pull controls are only shown when pull mode is enabled.
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
# Shift schedule disabled for continuous 3-week production
# If shift_schedule is empty or not present, initialize with empty list (shifts disabled)
if not cfg.get("shift_schedule"):
    cfg["shift_schedule"] = []

# Shift configuration UI (disabled - shifts not used in current setup)
if len(cfg.get("shift_schedule", [])) > 0:
    d1, d2 = st.columns(2)
    with d1:
        shift_start_min = st.number_input(
            "Shift start (minutes in day)",
            min_value=0,
            max_value=24 * 60,
            value=int(cfg["shift_schedule"][0].get("start_minute", 480))
        )
    with d2:
        shift_end_min = st.number_input(
            "Shift end (minutes in day)",
            min_value=0,
            max_value=24 * 60,
            value=int(cfg["shift_schedule"][0].get("end_minute", 960))
        )
    cfg["shift_schedule"][0]["start_minute"] = int(shift_start_min)
    cfg["shift_schedule"][0]["end_minute"] = int(shift_end_min)
else:
    # Shifts disabled - production runs continuously (24/7)
    st.info("ℹ️ **Shifts disabled** - Production runs continuously (24/7) for 3-week simulation.")
    cfg["shift_schedule"] = []


st.markdown("---")
st.subheader("Processing & Quality")
exp = st.expander("Per-stage timing, transport, defects", expanded=False)

def stage_controls(label: str, sid: str):
    st.markdown(f"**{label} ({sid})**")
    s = stage_by_id(cfg, sid)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        base = st.number_input(
            f"{sid} base_time/worker (min)",
            min_value=0.0,
            value=float(s.get("base_process_time_min", s.get("base_process_time_sec", 1.0) / 60.0)),
            step=0.5,
            key=f"{sid}_base"
        )
        s["base_process_time_min"] = float(base)
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
            f"{sid} transport (min)",
            min_value=0.0,
            value=float(s.get("transport_time_min", s.get("transport_time_sec", 0.0) / 60.0)),
            step=0.1,
            key=f"{sid}_tt"
        )
        s["transport_time_min"] = float(trans)
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
    st.markdown("**S2 per-output transport (min)**")
    s2 = stage_by_id(cfg, "S2")
    if "transport_time_to_outputs_min" not in s2 or s2.get("transport_time_to_outputs_min") is None:
        # Support both old and new field names
        old_dict = s2.get("transport_time_to_outputs_sec", {})
        base = float(s2.get("transport_time_min", s2.get("transport_time_sec", 0.0) / 60.0))
        if old_dict:
            s2["transport_time_to_outputs_min"] = {k: float(v) / 60.0 if isinstance(v, (int, float)) else v for k, v in old_dict.items()}
        else:
            s2["transport_time_to_outputs_min"] = {"C1": base, "C2": base, "C3": base}

    ttc1, ttc2, ttc3 = st.columns(3)
    with ttc1:
        s2_c1 = st.number_input("S2 → C1", min_value=0.0, value=float(s2["transport_time_to_outputs_min"].get("C1", 0.0)), step=0.1, key="S2_to_C1")
    with ttc2:
        s2_c2 = st.number_input("S2 → C2", min_value=0.0, value=float(s2["transport_time_to_outputs_min"].get("C2", 0.0)), step=0.1, key="S2_to_C2")
    with ttc3:
        s2_c3 = st.number_input("S2 → C3", min_value=0.0, value=float(s2["transport_time_to_outputs_min"].get("C3", 0.0)), step=0.1, key="S2_to_C3")

    s2["transport_time_to_outputs_min"] = {"C1": float(s2_c1), "C2": float(s2_c2), "C3": float(s2_c3)}

    stage_controls("Axis Assembly", "S3")
    stage_controls("Chassis Assembly", "S4")
    stage_controls("Final Assembly", "S5")





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
        # Push: keep demand-based settings already written in the sidebar
        cfg["parameters"]["release_stage_ids"] = cfg["parameters"].get("release_stage_ids", ["S1"])

    # Build env
    env = LegoLeanEnv(cfg, time_unit="min", seed=(None if seed is None else int(seed)))

    # Order release strategy differs by mode:
    # - Pull mode: Manually release orders based on user input (model_quantities)
    # - Push mode: Orders are automatically released based on demand forecast (queued immediately at start)
    if enable_pull:
        # Pull-mode: initial release only (then closed-loop CONWIP replenishes if enabled)
        # Release orders for each model type (FIFO queueing)
        total_released = 0
        release_order = []  # Track order for logging
        
        # Release orders in the order they appear in model_quantities
        # This ensures FIFO: first model enqueued will be processed first
        for model_id, qty in model_quantities.items():
            if qty > 0:
                env.enqueue_orders(qty=int(qty), model_id=model_id)
                total_released += int(qty)
                model_name = cfg.get("models", {}).get(model_id, {}).get("name", model_id)
                release_order.append(f"{qty}x {model_name}")
        
        if total_released == 0:
            st.warning("⚠️ No orders were released. Please specify quantities for at least one model.")
            st.stop()
        else:
            st.info(f"✅ Released {total_released} order(s) in FIFO order: {', '.join(release_order)}")
    else:
        # Push-mode: Forecast → Queue → Produce → Compare
        # 1. Forecast 3 weeks demand (generated during env init)
        # 2. Queue all orders immediately based on forecast
        # 3. Produce everything until finished
        # 4. Compare actual demand vs forecast in KPIs
        # In push mode, forecast automatically triggers production (auto-release always enabled)
        planned_total = sum(env.planned_release_qty.values()) if isinstance(env.planned_release_qty, dict) else 0
        if planned_total > 0:
            st.info(f"📊 Push mode: {planned_total} orders queued from forecast. Production will run until completion.")
        else:
            st.warning("⚠️ Push mode enabled but no orders scheduled. Check forecast parameters.")

    #2026-01-01 存疑，如果是push的话enqueue激活就够了，这样是否会反复激活S1？
    # Kick off: try starting all stages once
    for s in env.stages.values():
        env._push_event(env.t, "try_start", {"stage_id": s.stage_id})

    env.run_for(float(sim_time), max_events=1_000_000)
    t1 = time.time()

    st.success(f"Simulation finished in {t1 - t0:.3f} sec (wall).")

    kpis = env.get_kpis()
    st.subheader("KPIs")
    
    # Show per-model production KPIs if available (push mode)
    if not enable_pull and kpis.get("planned_release_qty_by_model"):
        st.markdown("### Production KPIs (Per-Model)")
        realized_by_model = kpis.get("demand_realized_by_model", {})
        planned_by_model = kpis.get("planned_release_qty_by_model", {})
        finished_by_model = kpis.get("finished_goods_by_model", {})
        
        if planned_by_model:
            kpi_data = []
            for model_id in sorted(planned_by_model.keys()):
                model_name = cfg.get("models", {}).get(model_id, {}).get("name", model_id)
                planned = planned_by_model.get(model_id, 0)
                realized = realized_by_model.get(model_id, 0)
                produced = finished_by_model.get(model_id, 0)
                unmet = abs(realized - produced)

                # Determine status based on production vs demand
                if realized > produced:
                    status = "⚠️ Unmet"
                elif realized < produced:
                    status = "⚠️ Overproduced"
                else:
                    status = "✅ Met"

                kpi_data.append({
                    "Model": f"{model_name} ({model_id})",
                    "Planned": planned,
                    "Realized Demand": realized,
                    "Produced": produced,
                    "Unmet Demand": unmet,
                    "Status": status
                })
            
            if kpi_data:
                kpi_df = pd.DataFrame(kpi_data)
                st.dataframe(kpi_df, use_container_width=True)
                
                # Summary totals
                total_planned = kpis.get("planned_release_qty", 0)
                total_realized = kpis.get("demand_realized_total", 0)
                total_produced = kpis.get("finished_units", 0)
                total_unmet = abs(total_realized - total_produced)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Planned", total_planned)
                with col2:
                    st.metric("Total Realized Demand", total_realized)
                with col3:
                    st.metric("Total Produced", total_produced)
                with col4:
                    st.metric("Unmet Demand", total_unmet, delta=f"{(total_unmet/total_realized*100) if total_realized > 0 else 0:.1f}%")
    
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
