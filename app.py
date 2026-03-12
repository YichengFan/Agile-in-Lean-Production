# app.py
# Streamlit control panel for LEGO Lean DES (Pull mode: CONWIP + Kanban) with KPI charts.

import copy
import json
import time
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px
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
        # 强制锁定 S1，
        cfg["parameters"]["release_stage_ids"] = ["S1"]
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
        # 2026-3-11margin_per_unit = st.number_input(
        #     "Margin per unit",
        #     min_value=0.0,
        #     value=float(cfg["parameters"].get("cost", {}).get("margin_per_unit", 5600.0)),
        #     step=100.0
        # )
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
       # cfg["parameters"]["cost"]["margin_per_unit"] = float(margin_per_unit)
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

    # Order release configuration: only visible when pull is enabled (box ticked).
    # In push mode (box not ticked), this section is hidden; release is forecast-based.
    if enable_pull:
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

                #2026-3-10 自动计算并显示 Target Takt ===
                calculated_takt = float(sim_time) / float(total_orders)
                st.metric("Calculated Target Takt (min/unit)", f"{calculated_takt:.2f}",
                          help="Simulation Time / Total Orders")
                cfg["parameters"]["target_takt_min"] = calculated_takt

            else:
                st.warning("⚠️ No orders specified. Please set quantity for at least one model.")
                cfg["parameters"]["target_takt_min"] = None  # 没有订单就不算
        else:
            st.info("No model definitions found in config. Using default materials.")
            # Fallback: single quantity input for backward compatibility
            initial_release = st.number_input("Initial release qty", min_value=0, value=12, step=1)
            if initial_release > 0:
                model_quantities["default"] = initial_release

        show_logs_n = st.number_input("Show last N logs", min_value=0, value=30, step=5)
    else:
        # Push mode: section hidden; release is forecast-based. Use defaults for run options.
        model_quantities = {}
        show_logs_n = 30
        cfg["parameters"]["target_takt_min"] = None  #2026-3-10 Push 模式下清空 Takt ===


st.subheader("Global Parameters")
sample_dt = st.number_input(
    "Timeline sample Δt (min)",
    min_value=0.5,
    value=float(cfg["parameters"].get("timeline_sample_dt_min", cfg["parameters"].get("timeline_sample_dt_sec", 5.0) / 60.0)),
    step=0.5,
    help="Interval for logging time-series data to charts"
)
cfg["parameters"]["timeline_sample_dt_min"] = float(sample_dt)
# --- 新增：全局财务参数 ---
st.markdown("**Financial Parameters**")
f1, f2 = st.columns(2)
with f1:
    unit_price = st.number_input(
        "Unit Price (Revenue)",
        min_value=0.0,
        value=float(cfg["parameters"].get("cost", {}).get("unit_price", 10000.0)),
        step=100.0,
        help="Revenue per finished unit"
    )
with f2:
    unit_material_cost = st.number_input(
        "Unit Material Cost",
        min_value=0.0,
        value=float(cfg["parameters"].get("cost", {}).get("unit_material_cost", 4400.0)),
        step=100.0,
        help="Material cost per unit"
    )

# 自动计算 Margin 并在界面上提示
calculated_margin = max(0.0, float(unit_price) - float(unit_material_cost))
st.info(f"💡 Calculated Margin per unit: **{calculated_margin:.2f}**")

# 写入全局配置
cfg["parameters"].setdefault("cost", {})
cfg["parameters"]["cost"]["unit_price"] = float(unit_price)
cfg["parameters"]["cost"]["unit_material_cost"] = float(unit_material_cost)
cfg["parameters"]["cost"]["margin_per_unit"] = float(calculated_margin)


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
        # --- 开始替换 ---

        # 1. 注入 CSS：彻底隐藏所有图标，并禁止鼠标点击
        st.markdown(
            """
            <style>
            /* 针对 Release Stage 区域：禁止鼠标交互 */
            div[data-testid="stMultiSelect"] {
                pointer-events: none;
            }
            /* 隐藏所有 SVG 图标（叉号、下拉箭头） */
            div[data-testid="stMultiSelect"] svg {
                display: none !important;
            }
            /* 隐藏输入框，防止出现光标 */
            div[data-testid="stMultiSelect"] input {
                display: none !important;
            }
            /* 隐藏右侧的下拉点击区域 */
            div[data-testid="stMultiSelect"] div[role="button"] {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        def enforce_s1_only():
            st.session_state.locked_release_stage = ["S1"]
        if "locked_release_stage" not in st.session_state:
            st.session_state.locked_release_stage = ["S1"]

        release_stage_ids = st.multiselect(
            "Release stage(s)",
            options=["S1"],
            key="locked_release_stage",
            on_change=enforce_s1_only,
            help="Fixed to S1 for this simulation."
        )
        cfg["parameters"]["release_stage_ids"] = release_stage_ids
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

    # Assembly trace toggle
    trace_assembly = st.sidebar.checkbox("Trace assembly (store per-job consumed/produced)", value=False)
    cfg["parameters"]["trace_assembly"] = bool(trace_assembly)
    # Apply mode settings
    if enable_pull:
        cfg["parameters"]["release_stage_ids"] = release_stage_ids
        cfg["parameters"]["conwip_wip_cap"] = int(conwip_cap) if int(conwip_cap) > 0 else None
        cfg["parameters"]["auto_release_conwip"] = bool(auto_release)
        # 改为引用主面板定义的 cap_c3, cap_d1, cap_d2
        cfg["parameters"]["kanban_caps"] = {"C3": int(cap_c3), "D1": int(cap_d1), "D2": int(cap_d2)}
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

    # Show per-model production KPIs if available (push mode)
    if not enable_pull and kpis.get("planned_release_qty_by_model"):
        st.markdown("### Production Plan vs. Reality (Per-Model)")
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
                unmet_m = max(0, realized - produced)
                overproduced_m = max(0, produced - realized)
                total_difference = unmet_m + overproduced_m

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
                    "Total Difference": total_difference,
                    "Status": status
                })

            if kpi_data:
                kpi_df = pd.DataFrame(kpi_data)
                st.dataframe(kpi_df, use_container_width=True)

                total_planned = kpis.get("planned_release_qty", 0)
                total_realized = kpis.get("demand_realized_total", 0)
                total_produced = kpis.get("finished_units", 0)
                # Total difference = sum over models of (unmet + overproduced) per model
                total_difference = sum(
                    max(0, realized_by_model.get(m, 0) - finished_by_model.get(m, 0))
                    + max(0, finished_by_model.get(m, 0) - realized_by_model.get(m, 0))
                    for m in planned_by_model.keys()
                )

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Planned", total_planned)
                with col2:
                    st.metric("Total Realized Demand", total_realized)
                with col3:
                    st.metric("Total Produced", total_produced)
                with col4:
                    st.metric("Total Difference", total_difference,
                              delta=f"{(total_difference / total_realized * 100) if total_realized > 0 else 0:.1f}%",
                              delta_color="inverse")
    # =====================================================================
    # 🌟 1. Executive Dashboard (Top Metrics)
    # =====================================================================
    st.markdown("---")
    st.subheader("🌟 Executive Dashboard")

    # Row 1: Core Financial & Flow Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Net Profit", f"€{kpis.get('profit', 0):,.2f}")
    with m2:
        st.metric("Service Level", f"{kpis.get('availability', 0) * 100:.1f}%")
    with m3:
        st.metric("Avg Lead Time", f"{kpis.get('lead_time_avg_min', 0):.1f} min")
    with m4:
        st.metric("Avg WIP", f"{kpis.get('wip_avg_units', 0):.1f} units")

    # Row 2: Flow Pacing (Target vs Actual)
    st.markdown("#### 🏃‍♂️ Flow Pacing")

    raw_takt = cfg["parameters"].get("target_takt_min")
    actual_cycle = float(kpis.get("cycle_time_avg_min", 0.0))

    # 如果有 Target Takt (Pull 模式)
    if raw_takt is not None:
        takt_target = float(raw_takt)
        delta_val = takt_target - actual_cycle

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Target Takt", f"{takt_target:.1f} min/unit",
                      help="The pace required to meet customer demand.")
        with c2:
            st.metric("Actual Cycle Time", f"{actual_cycle:.1f} min/unit",
                      delta=f"{delta_val:.1f} min (vs Target)",
                      delta_color="normal",
                      help="Average time between consecutive finished products.")
        with c3:
            if actual_cycle == 0:
                st.info("ℹ️ Not enough products finished to calculate cycle time.")
            elif actual_cycle > takt_target:
                st.error(
                    "⚠️ **Too Slow!** The line cannot meet the customer takt. Check for bottlenecks (Blocking/Starvation) in diagnostics.")
            elif actual_cycle < takt_target * 0.7:
                st.warning(
                    "⚠️ **Too Fast!** Producing much faster than required takt. This may cause overproduction waste if not strictly controlled by Kanban.")
            else:
                st.success(
                    "✅ **Perfect Pace!** Actual cycle time closely matches customer takt. The Lean flow is healthy.")

    # 如果没有 Target Takt (Push 模式)
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Actual Cycle Time", f"{actual_cycle:.1f} min/unit",
                      help="Average time between consecutive finished products.")
        with c2:
            if actual_cycle == 0:
                st.info("ℹ️ Not enough products finished to calculate cycle time.")
            else:
                st.info(
                    "ℹ️ **Push Mode**: Flow pace is determined by forecast release rate. No fixed Target Takt applied.")
    # =====================================================================
    # 💰 2. Financial & Cost Analysis
    # =====================================================================
    st.markdown("---")
    st.subheader("💰 Financial & Cost Analysis")
    f1, f2 = st.columns([1, 1])

    with f1:
        st.markdown("**Cost Breakdown (Lean Perspective)**")

        # 获取基础数据
        total_mat = kpis.get('cost_material', 0)
        total_lab = kpis.get('cost_labor', 0)
        inv_holding = kpis.get('cost_inventory', 0)
        waste_cost = kpis.get('overproduction_waste_cost', 0)

        # 在精益视角下，把白白浪费的材料和人工从“有效成本”里剔除出来，单独展示
        # 先算出当前模拟下，真实的材料和人工比例
        total_base_cost = total_mat + total_lab
        mat_ratio = total_mat / total_base_cost  # 如果材料占大头，算出来可能是 0.791
        lab_ratio = total_lab / total_base_cost  # 算出来可能是 0.209

        # 然后再用真实的比例去扣除
        effective_mat = max(0, total_mat - (waste_cost * mat_ratio))
        effective_lab = max(0, total_lab - (waste_cost * lab_ratio))

        cost_data = {
            "Category": [
                "Effective Material",
                "Effective Labor",
                "Inventory cost (Rent/Space)",
                "⚠️ Overproduction Waste (Dead Stock)"
            ],
            "Amount": [
                effective_mat,
                effective_lab,
                inv_holding,
                waste_cost
            ]
        }
        df_cost = pd.DataFrame(cost_data)

        # Try to use Plotly for the Pie Chart, fallback to standard bar chart if not installed
        try:
            import plotly.express as px

            # 给浪费分配一个醒目的红色
            color_discrete_map = {
                "Effective Material": "#1f77b4",
                "Effective Labor": "#9edae5",
                "Holding (Rent/Space)": "#ff7f0e",
                "⚠️ Overproduction Waste (Dead Stock)": "#d62728"
            }
            fig_pie = px.pie(df_cost, values='Amount', names='Category', hole=0.3,
                             color='Category', color_discrete_map=color_discrete_map)
            fig_pie.update_traces(textposition='auto', textinfo='percent')

            # 把图例放到下面以免挤占空间
            fig_pie.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5)
            )

            st.plotly_chart(fig_pie, use_container_width=True)
        except ImportError:
            st.info("💡 Tip: Install `plotly` (`pip install plotly`) to view this as a Pie Chart.")
            st.bar_chart(df_cost.set_index("Category"))

    with f2:
        st.markdown("**Profit & Loss Breakdown**")
        st.write(f"- **Total Revenue**: €{kpis.get('revenue_total', 0):,.2f}")
        st.write(f"- **Total Cost**: €{kpis.get('cost_total', 0):,.2f}")
        st.info(f"**Net Profit**: €{kpis.get('profit', 0):,.2f}")

        # Highlight lean wastes/losses if they exist
        opp_loss = kpis.get('revenue_opportunity_loss', 0)
        waste_cost = kpis.get('overproduction_waste_cost', 0)

        if opp_loss > 0:
            st.warning(f"⚠️ **Opportunity Loss (Unmet Demand)**: €{opp_loss:,.2f}")
        if waste_cost > 0:
            st.error(f"🗑️ **Overproduction Waste Cost**: €{waste_cost:,.2f}")

    # =====================================================================
    # 📊 3. Production Flow Charts
    # =====================================================================
    st.markdown("---")
    st.subheader("📊 Production Flow")

    # Team utilization
    st.markdown("**Team Utilization (%)**")
    util_data = {k: v * 100 for k, v in kpis.get("utilization_per_team", {}).items()}
    if util_data:
        st.bar_chart(pd.Series(util_data))

    if env.timeline:
        df = pd.DataFrame(env.timeline).set_index("t")
        st.markdown("**WIP and Finished Units**")
        st.line_chart(df[["wip", "finished"]], height=220)

        # Throughput per day: x = day (1 unit = 480 min), y = units produced in that day
        MIN_PER_DAY = 480.0  # 8-hour working day
        df_day = df.copy()
        df_day["day"] = (df_day.index / MIN_PER_DAY).astype(int)
        daily = df_day.groupby("day").agg(
            finished_first=("finished", "first"),
            finished_last=("finished", "last"),
        )
        daily["throughput_per_day"] = daily["finished_last"] - daily["finished_first"]
        st.markdown("**Throughput (units/day)**")
        st.line_chart(daily[["throughput_per_day"]], height=220)

        buffer_cols = [c for c in ["B", "C1", "C2", "C3", "D1", "D2", "E"] if c in df.columns]
        if buffer_cols:
            st.markdown("**Buffer Levels**")
            st.line_chart(df[buffer_cols], height=260)
    else:
        st.info("No timeline captured. Increase simulation time or decrease sample interval.")

    # =====================================================================
    # 🛠️ 4. Diagnostics & Bottlenecks
    # =====================================================================
    st.markdown("---")
    st.subheader("🛠️ Diagnostics & Bottlenecks")

    with st.expander("🔍 View Bottleneck Data (Blocking & Starvation)"):
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            st.markdown("**Blocking Counts**")
            st.write(kpis.get("blocking_counts", {}))
            if hasattr(env, "kanban_blocking_counts"):
                st.markdown("**Kanban Blocking Counts**")
                st.write(getattr(env, "kanban_blocking_counts"))
        with col_diag2:
            st.markdown("**Starvation Counts**")
            st.write(kpis.get("starvation_counts", {}))

    with st.expander("⚙️ View Pull Diagnostics Configuration"):
        diag = {
            "conwip_wip_cap": cfg["parameters"].get("conwip_wip_cap"),
            "auto_release_conwip": cfg["parameters"].get("auto_release_conwip"),
            "release_stage_ids": cfg["parameters"].get("release_stage_ids"),
            "kanban_caps": cfg["parameters"].get("kanban_caps"),
        }
        st.json(diag)

    with st.expander("📄 View Raw KPI JSON"):
        st.json(kpis)

    # Downloads
    # ... (Keep your existing Downloads section here) ...
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
