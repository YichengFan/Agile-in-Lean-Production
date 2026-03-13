# LEGO Lean Production — Simulation Model (Current Version)

**Purpose:** This document is the **authoritative specification** of the discrete-event simulation (DES): system structure, Push/Pull modes, event logic, parameters, KPIs, and configuration. It is intended for readers who need to understand, validate, or extend the simulator.

**Related docs:** Multi-model configuration and usage → [MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md). Mathematical formulation → [mathematical_model.md](mathematical_model.md).

---

## Quick Start

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Run the interactive UI (Streamlit)

```bash
streamlit run app.py
```

3) Or run the CLI example

```bash
python env.py
```

**Notes:**
- The system supports **multi-model production** (e.g. m01–m04) and **Push** or **Pull** mode from the UI
- **Pull mode** (Enable Pull ticked): manual **Order Release Configuration** (quantities per model), CONWIP WIP cap, Kanban caps, simulation time adjustable
- **Push mode** (Enable Pull unticked): demand-forecast-based release; Order Release section is hidden; simulation time follows demand horizon
- Time units are **minutes**; configuration is in `env.py`'s `CONFIG` dictionary
- Optional random seed in the UI for reproducibility

---

## Mathematical Model

For the **mathematical model** (objective function, decision variables, constraints, cost formulations), please refer to [`mathematical_model.md`](mathematical_model.md).

This document focuses on the **simulation model** implementation details and current system capabilities.

---

## Table of Contents

1. [System Overview](#1-system-overview)  
2. [Sets, Indices, Graph](#2-sets-indices-graph)  
3. [Parameters](#3-parameters)  
4. [Demand Forecasting System](#4-demand-forecasting-system)  
5. [State Variables](#5-state-variables)  
6. [Event Logic](#6-event-logic)  
7. [KPIs](#7-kpis)  
8. [Configuration](#8-configuration)  
9. [Simulation Implementation Mapping](#9-simulation-implementation-mapping)  
10. [Simulation Assumptions](#10-simulation-assumptions)  
11. [Key Features](#11-key-features)  
12. [Usage Examples](#12-usage-examples)  
13. [Future Work](#13-future-work--todos)  
14. [File Structure](#14-file-structure)  
15. [Getting Help](#15-getting-help)  
16. [Validation Targets](#16-validation-targets)

---

# Simulation Model (Discrete-Event Simulation)

## 1) System Overview

The system implements a **dual-mode production simulator** (Push and Pull) with **multi-model production**:

- **Push mode**: demand forecasting over a horizon (e.g. 3 weeks), automatic order release, per-model planned vs. realized vs. produced KPIs, unmet demand and overproduction
- **Pull mode**: token-gated order release at S1, CONWIP WIP cap, Kanban buffer caps, manual Order Release Configuration (quantities per model) in the UI when “Enable Pull” is ticked
- Both modes share the same DES engine, BOM-driven stages, defects/rework logic, and KPI framework

### Production Modes

**Push Mode (Demand-based):**
- Demand horizon (weeks), weekly demand mean, forecast and realization noise, procurement waste
- Orders are queued at simulation start from the forecast; no manual order quantities in the UI
- Tracks planned vs. realized vs. produced per model; KPIs for unmet demand and overproduction
- Simulation time is derived from the demand horizon (e.g. weeks × 5 days × 8 hours × 60 min)

**Pull Mode (CONWIP + Kanban):**
- **Order Release Configuration** (sidebar): visible only when “Enable Pull” is ticked; user sets quantity per model to release
- Token-gated release at release stages (e.g. S1); rework jobs bypass token gating (`is_rework=True`)
- CONWIP WIP cap and closed-loop CONWIP (auto release on finish) option
- Kanban caps on output buffers; simulation time is user-configurable (minutes)

---

## 2) Sets, Indices, Graph

- **Stages (processing nodes)**: S<sub>1–5</sub> = {S<sub>1</sub>, S<sub>2</sub>, S<sub>3</sub>, S<sub>4</sub>, S<sub>5</sub>}
  - S<sub>1</sub>: Type Sorting  
  - S<sub>2</sub>: Set Sorting (Kit Build)
  - S<sub>3</sub>: Axis Assembly  
  - S<sub>4</sub>: Chassis Assembly  
  - S<sub>5</sub>: Final Assembly
- **Buffers (inventories)**: B = {A, B, C<sub>1</sub>, C<sub>2</sub>, C<sub>3</sub>, D<sub>1</sub>, D<sub>2</sub>, E}
  - A: Raw materials buffer
  - B: Sorted parts buffer
  - C<sub>1</sub>, C<sub>2</sub>, C<sub>3</sub>: Kit buffers
  - D<sub>1</sub>, D<sub>2</sub>: Subassembly buffers
  - E: Finished goods buffer
- **Teams (resources)**: T = {T<sub>1</sub>, T<sub>2</sub>, T<sub>3</sub>, T<sub>4</sub>, T<sub>5</sub>}
- **Product Models**: M = {m01, m02, m03, m04}
  - m01: Sly_Slider
  - m02: Gliderlinski
  - m03: Icky_Ice_Glider
  - m04: Icomat_2000X
- **Arcs**: directed edges between buffers and stages describing the flow:  
  S<sub>1</sub> → B → S<sub>2</sub> → {C<sub>1</sub>, C<sub>2</sub>, C<sub>3</sub>} → (S<sub>3</sub>, S<sub>4</sub>, S<sub>5</sub>) → E

---

## 3) Parameters

### Time Parameters

- **Time unit**: **Minutes** (changed from seconds for realism)
- **Base processing time** at stage *i*: τ<sub>i</sub> > 0 (minutes)
- **Number of workers** at stage *i*: w<sub>i</sub> ∈ ℕ<sup>+</sup>
- **Effective processing time**: τ<sub>i</sub> / √w<sub>i</sub> (deterministic square root efficiency)
- **Transport time**: δ<sub>i</sub> ≥ 0 (minutes), constant per stage (and per-output for S2→C1/C2/C3)

### Worker Efficiency

The system uses a **deterministic square root efficiency function**:

```
effective_time = base_time / √(workers)
```

This means:
- 1 worker: effective_time = base_time / 1 = base_time
- 2 workers: effective_time = base_time / √2 ≈ base_time / 1.414
- 4 workers: effective_time = base_time / 2

**Example**: With τ<sub>i</sub> = 60 minutes, δ<sub>i</sub> = 5 minutes:
- 1 worker: E[S<sub>i</sub>] ≈ 60 / 1 + 5 = 65 minutes
- 2 workers: E[S<sub>i</sub>] ≈ 60 / 1.414 + 5 ≈ 47 minutes
- 4 workers: E[S<sub>i</sub>] ≈ 60 / 2 + 5 = 35 minutes

### Demand Forecasting Parameters (Push Mode)

- **Demand horizon**: H weeks (default: 3 weeks)
- **Weekly demand mean**: μ<sub>weekly</sub> (total units per week)
- **Model demand probabilities**: P(m01), P(m02), P(m03), P(m04)
  - Default: m01=60%, m02=10%, m03=20%, m04=10%
- **Forecast noise**: ε<sub>f</sub> ∈ [0, 1] (percentage applied to forecast generation)
- **Realization noise**: ε<sub>r</sub> ∈ [0, 1] (percentage applied to actual demand)
- **Procurement waste rate**: w ∈ [0, 1] (safety stock/waste percentage)

### Model-Specific Parameters

- **Model-specific BOMs**: Each stage can have different material requirements per model
  - `required_materials_by_model`: Dict[model_id, Dict[item_id, quantity]]
- **Model-specific outputs**: Each model can produce different finished goods
  - `output_buffers_by_model`: Dict[model_id, Dict[buffer_id, Dict[item_id, quantity]]]

### Defect and Rework Parameters

- **Defect probability**: q<sub>i</sub> ∈ [0, 1] (default: 0.0 for all stages)
- **Rework stage**: `rework_stage_id` per stage (typically the same stage, e.g. S2 → S2)
- **Rework semantics**: Each job is tagged with `is_rework` (False = fresh, True = already reworked). On defect:
  - **Fresh batch** (`is_rework` False): send to rework (one `try_start` at `rework_stage_id` with `is_rework=True`); no release token consumed
  - **Already rework** (`is_rework` True): scrap (decrement WIP, drop one release time); no second rework
- Applies in both Push and Pull modes

### Other Parameters

- **Buffer capacity**: cap<sub>b</sub> ∈ ℕ ∪ {∞}
- **Production schedule**: 24/7 (no shift windows; simulation runs continuously)

---

## 4) Demand Forecasting System

### Forecast Generation Process

1. **Total Forecast Calculation**:
   - Generate total forecast for H weeks: F<sub>total</sub> = H × μ<sub>weekly</sub> × (1 ± ε<sub>f</sub>)

2. **Model Allocation**:
   - Allocate forecast across models based on probability distribution:
     - F<sub>m01</sub> = F<sub>total</sub> × P(m01)
     - F<sub>m02</sub> = F<sub>total</sub> × P(m02)
     - etc.

3. **Realization**:
   - Apply realization noise to each model's forecast:
     - D<sub>realized</sub> = F<sub>model</sub> × (1 ± ε<sub>r</sub>)

4. **Production Planning**:
   - Planned production = Forecast (before realization noise)
   - Actual demand = Realized demand (after realization noise)

5. **Material Procurement**:
   - Calculate total materials needed for all planned orders
   - Add procurement waste: Materials = Total × (1 + w)
   - Add materials to Buffer A at simulation start

6. **Order Release**:
   - All orders are queued immediately at simulation start (push_auto_release = True)
   - Orders are processed in FIFO order per model

### Key Variables

- `planned_release_qty_by_model`: Planned production quantity per model (based on forecast)
- `demand_realized_by_model`: Actual realized demand per model (with noise)
- `finished_goods_by_model`: Actually produced quantity per model

### Performance Metrics

- **Unmet Demand**: |realized - produced| (absolute difference)
- **Status Indicators**:
  - ✅ **Met**: produced = realized
  - ⚠️ **Unmet**: produced < realized
  - ⚠️ **Overproduced**: produced > realized

---

## 5) State Variables

- **Buffer levels**: X<sub>b</sub>(t) ∈ [0, cap<sub>b</sub>] (itemized by item_id)
- **Stage busy**: Y<sub>i</sub>(t) ∈ {0, 1}
- **Team busy**: U<sub>i</sub>(t) (utilization tracking)
- **WIP**: L(t) = number of released but unfinished jobs
- **Finished goods**: C(t) = finished goods in E (tracked per model)
- **Order queue**: Q<sub>model</sub>(t) = FIFO queue of orders per model

---

## 6) Event Logic

### Order Release

**Push mode:**
- Orders are queued at simulation start from the demand forecast; each order has a `model_id`
- Stored in `_stage_order_models[stage_id]` (FIFO); no manual order entry in the UI

**Pull mode:**
- User sets **Order Release Configuration** (quantity per model) in the sidebar when “Enable Pull” is ticked
- Those quantities are enqueued at run time as orders (FIFO); release stages consume tokens and pop model_id from the queue
- Rework jobs do not consume tokens (`is_rework=True`)

### Try-Start at Stage i

A stage i can start only if all of the following hold:

1. **Idle condition**: Y<sub>i</sub>(t) = 0 (`stage.busy == False`)
2. **Release-stage token gating** (if i ∈ release_stage_ids and **not** rework):
   - For **rework** jobs (`is_rework=True`), token gating is **bypassed** (no token consumed)
   - For normal jobs: `stage_orders[i] > 0`, then one token consumed and model_id popped from FIFO
3. **Material availability (BOM)**:
   - For release stages: model_id from order queue or event payload
   - For downstream stages: model_id inferred from available materials
   - All required materials must be available in input buffers
4. **Kanban gating** (if enabled): output buffers must not exceed Kanban caps

If all checks pass:
- Pull BOM items from buffers (model-specific)
- Set Y<sub>i</sub>(t) = 1
- Calculate processing time: `base_time / √workers`
- Schedule `complete` event at t + S<sub>i</sub> with **`is_rework`** set from the try_start payload (fresh vs rework)

### Completion at Stage i

On `complete(i)`:
- Stop team utilization timer U<sub>i</sub>
- Read **`is_rework`** from the complete event payload (set at try_start).
- **Defect handling** (if q<sub>i</sub> > 0 and defect sampled):
  - **If `is_rework` True**: scrap (decrement WIP, drop one release time from FIFO); no delivery; free stage; schedule one normal try_start so the stage can pull the next job; then **return** (no output push, no “wake consumers”).
  - **If `is_rework` False** and rework_stage_id set: push one **try_start** at `rework_stage_id` with `is_rework=True` and same model_id; no delivery; free stage; do **not** schedule a normal try_start (rework try_start was already pushed); then **return**.
  - **If `is_rework` False** and no rework stage: scrap (same as above); free stage; schedule normal try_start; return.
- **If no defect**: push outputs (model-specific), free stage, schedule normal try_start, wake downstream consumers. Use `output_buffers_by_model[model_id]` if available, else `output_buffers`. If output blocked, retry complete later (payload preserves `is_rework`). If output goes to finished buffer E, increment `finished_goods_by_model[model_id]` and decrement WIP.

### Model ID Propagation

- **Release stages**: model_id comes from the order queue
- **Downstream stages**: model_id is inferred by matching available materials to `required_materials_by_model`
- If inference fails, uses default model_id

---

## 7) KPIs

### Production KPIs

- **Throughput**: θ = C(t) / t (finished units per minute)
- **Average WIP**: L̄ = (1 / t) ∫₀ᵗ L(s) ds
- **Average lead time**: W̄ (calculated from release to completion timestamps)
- **Average cycle time**: average time between consecutive finished units (flow pacing indicator)
- **Utilization**: ρ<sub>i</sub> = U<sub>i</sub>(t) / t (per team)
- **Service level**: SL = N<sub>fin</sub>(t) / N<sub>target</sub>

### Demand & Production KPIs (Per Model)

- **Planned Release Quantity**: Forecast-based production plan per model
- **Realized Demand**: Actual customer demand per model (with noise)
- **Produced Quantity**: Actually finished units per model
- **Sales Units**: min(produced, realized) per model (used for revenue)
- **Unmet Demand Units**: max(0, realized − produced) per model
- **Overproduced Units**: max(0, produced − realized) per model
- **Status**: Met / Unmet / Overproduced (based on produced vs. realized)

### Quality KPIs

- **Defect rate per stage**: DR<sub>i</sub> = N<sub>i,def</sub> / (N<sub>i,def</sub> + N<sub>i,ok</sub>)
- **Total defect rate**: DR<sub>total</sub> = Σ N<sub>i,def</sub> / Σ (N<sub>i,def</sub> + N<sub>i,ok</sub>)

### Flow KPIs

- **Starvation counts**: Stage wants to start but lacks input materials
- **Blocking counts**: Stage finishes but downstream buffer is full
- **Kanban blocking counts**: Stage blocked by Kanban cap (if enabled)

### Cost & Profit KPIs

- **Revenue (`revenue_total`)**: `unit_price` × **sales_units** (so overproduction is *not* counted as sold)
- **Material cost (`cost_material`)**: depends on `material_cost_mode` (e.g., procurement based on planned forecast + waste)
- **Labor cost**: Σ (team busy time × team size × labor_rate)
- **Inventory holding cost (`cost_inventory`)**: Σ (∫inventory dt × holding_rate) **excluding Buffer A** (raw-material warehouse)
- **Profit (`profit`)**: Revenue − (Material + Labor + Inventory + Other)
- **Revenue opportunity loss (`revenue_opportunity_loss`)**: `margin_per_unit` × unmet_demand_units (foregone contribution margin)
- **Overproduction waste cost (`overproduction_waste_cost`)**: (unit_material_cost + avg_labor_cost_per_unit) × overproduced_units
  - *Diagnostic only*: do **not** add it on top of `cost_total`, because material/labor are already included there.

---

## 8) Configuration

### Main Configuration File

All configuration is in `env.py` under the `CONFIG` dictionary:

```python
CONFIG = {
    "buffers": [...],      # Buffer definitions
    "teams": [...],        # Team definitions
    "stages": [...],       # Stage definitions with model-specific BOMs
    "models": {...},       # Model definitions
    "parameters": {
        "push_demand_enabled": True,
        "push_demand_horizon_weeks": 3,
        "push_weekly_demand_mean": 25,
        "push_forecast_noise_pct": 0.2,
        "push_realization_noise_pct": 0.1,
        "push_procurement_waste_rate": 0.15,
        "model_demand_probabilities": {
            "m01": 0.6, "m02": 0.1, "m03": 0.2, "m04": 0.1
        },
        ...
    }
}
```

### Model-Specific BOM Configuration

Each stage can have model-specific materials:

```python
{
    "stage_id": "S1",
    "required_materials_by_model": {
        "m01": {"x01": 2, "x02": 4},
        "m02": {"x01": 3, "x02": 3},
        ...
    },
    "output_buffers_by_model": {
        "m01": {"B": {"item1": 1}},
        "m02": {"B": {"item2": 1}},
        ...
    }
}
```

### UI Configuration (app.py)

The Streamlit UI allows:
- **Mode**: Enable Pull (checkbox) — Pull shows Order Release Configuration and CONWIP/Kanban; Push shows demand horizon, forecast/noise, margin
- **Order Release Configuration**: visible only when Pull is enabled; quantity per model (m01–m04, etc.)
- **Simulation time**: in Pull mode user-defined (minutes); in Push mode derived from demand horizon
- **Global parameters**: timeline sample Δt (min) for charts. *Target takt* is not in CONFIG; in Pull mode the UI computes it as simulation time ÷ total orders and uses it only for the Flow Pacing comparison (Target vs Actual cycle time).
- **Pull controls** (when Pull enabled): CONWIP WIP cap, closed-loop CONWIP, release stage(s), Kanban caps
- **Processing & Quality** (expander): per-stage base time, workers, transport, defect rate
- **Initial buffer stocks**: JSON per buffer
- **Random seed**: optional for reproducibility

---

## 9) Simulation Implementation Mapping

| Simulation Element | `env.py` Implementation |
|-------------------|------------------------|
| τ<sub>i</sub> (processing time) | `base_process_time_min` with deterministic worker efficiency |
| w<sub>i</sub> | `workers` |
| Effective time | `base_time / √workers` (deterministic) |
| δ<sub>i</sub> | `transport_time_min` (or `transport_time_to_outputs_min` for S2 → C1/C2/C3) |
| q<sub>i</sub> | `defect_rate` (default: 0.0) |
| Rework stage | `rework_stage_id` (per stage; often same stage) |
| Fresh vs rework | `is_rework` on try_start/complete payloads; rework bypasses token gating |
| Model-specific BOM | `required_materials_by_model[model_id]` |
| Model-specific outputs | `output_buffers_by_model[model_id]` |
| Forecast parameters | `push_*` parameters in `CONFIG` |
| Model probabilities | `model_demand_probabilities` |
| Order queue | `_stage_order_models[stage_id]` (FIFO) |
| WIP L(t) | `self.current_wip` |
| Lead time tracking | `self._release_times` (deque) |
| Finished per model | `self.finished_goods_by_model` |
| Planned per model | `self.planned_release_qty` |
| Realized per model | `self.realized_demand_total` |

---

## 10) Simulation Assumptions

- **Discrete-event simulation**: Events processed in chronological order using a priority queue
- **Dual mode**: Push (forecast-based release) or Pull (token-gated + CONWIP + Kanban); same engine and BOM/defect logic
- **Time unit**: Minutes (realistic production times)
- **Worker efficiency**: Deterministic square root function (base_time / √workers)
- **24/7 production**: Shift schedules disabled for continuous operation
- **Model-specific BOMs**: Each model can have different material requirements
- **FIFO order processing**: Orders processed in first-in-first-out order per model
- **Defect/rework**: One rework per job (fresh → rework; rework again → scrap); `is_rework` carried on try_start and complete events; rework does not consume release tokens
- **Material procurement** (Push): Materials added to Buffer A at start based on forecast + waste

---

## 11) Key Features

### Multi-Model Production

- Support for 4 different product models (m01-m04)
- Model-specific BOMs at each stage
- Model-specific outputs
- Per-model KPI tracking
- Model ID propagation through production flow

### Demand Forecasting

- 3-week forecast generation
- Model probability distribution
- Forecast and realization noise
- Automatic order queuing
- Material procurement with waste/safety stock

### Defect and Rework

- Per-stage defect rate and rework_stage_id (typically same stage)
- `is_rework` on try_start and complete: fresh batch → one rework allowed; defect on rework → scrap
- Rework jobs bypass release-stage token gating; no double delivery or double try_start on defect path

### Realistic Time System

- Time in minutes (not seconds)
- Deterministic worker efficiency (√workers)
- Constant processing time per stage: effective_time = base_time / √workers (no random distributions)
- Transport times per stage (and per-output for S2 → C1/C2/C3)

### Fixed Configuration Parameters (env.py only)

These parameters are defined in `env.py` in the `CONFIG` dict. Values in the tables below are the defaults; many are overridable via the UI. To change any value that has no UI control, edit `env.py` directly.

#### Structural and control (CONFIG / parameters)

| Parameter | Value | Note |
|-----------|--------|------|
| `parameters.default_model_id` | `"m01"` | Default model when not specified |
| `parameters.timeline_sample_dt_min` | `5.0` | Timeline sampling interval (min) |
| `parameters.finished_buffer_ids` | `["E"]` | Buffers that count as finished goods |
| `parameters.release_stage_ids` | `["S1"]` (if omitted) | Token-gated release stages; UI assumes S1 |
| `parameters.supplier_stage_ids` | `[]` | Supplier stages (empty = S1 pulls from A normally) |
| `parameters.material_cost_mode` | `"procure_forecast"` | How material cost is computed (push) |

#### Models (CONFIG.models)

| Parameter | Value |
|-----------|--------|
| `models.m01.name` | Sly_Slider |
| `models.m02.name` | Gliderlinski |
| `models.m03.name` | Icky_Ice_Glider |
| `models.m04.name` | Icomat_2000X |

#### Model demand probabilities (push forecast allocation)

| Parameter | Value |
|-----------|--------|
| `parameters.model_demand_probabilities.m01` | 0.60 |
| `parameters.model_demand_probabilities.m02` | 0.10 |
| `parameters.model_demand_probabilities.m03` | 0.20 |
| `parameters.model_demand_probabilities.m04` | 0.10 |

#### Cost (parameters.cost)

| Parameter | Value |
|-----------|--------|
| `cost.unit_price` | 10000 |
| `cost.unit_material_cost` | 4400 |
| `cost.margin_per_unit` | 5600 |
| `cost.labor_costs_per_team_min.T1` | 0.25 |
| `cost.labor_costs_per_team_min.T2` | 0.25 |
| `cost.labor_costs_per_team_min.T3` | 0.25 |
| `cost.labor_costs_per_team_min.T4` | 0.25 |
| `cost.labor_costs_per_team_min.T5` | 0.25 |
| `cost.holding_costs_per_buffer_min.A` | 0.0100 |
| `cost.holding_costs_per_buffer_min.B` | 0.0010 |
| `cost.holding_costs_per_buffer_min.C1` | 0.0130 |
| `cost.holding_costs_per_buffer_min.C2` | 0.0210 |
| `cost.holding_costs_per_buffer_min.C3` | 0.0100 |
| `cost.holding_costs_per_buffer_min.D1` | 0.2600 |
| `cost.holding_costs_per_buffer_min.D2` | 0.4200 |
| `cost.holding_costs_per_buffer_min.E` | 0.7600 |
| `cost.demand_qty` | None |

#### Pull / CONWIP defaults (parameters; UI can override)

| Parameter | Value |
|-----------|--------|
| `parameters.conwip_wip_cap` | 12 |
| `parameters.auto_release_conwip` | True |
| `parameters.kanban_caps.C3` | 4 |
| `parameters.kanban_caps.D1` | 4 |
| `parameters.kanban_caps.D2` | 4 |

#### Push demand defaults (parameters; UI can override)

| Parameter | Value |
|-----------|--------|
| `parameters.push_demand_enabled` | True |
| `parameters.push_auto_release` | False |
| `parameters.push_demand_horizon_weeks` | 3 |
| `parameters.push_weekly_demand_mean` | 25 |
| `parameters.push_forecast_noise_pct` | 0.2 |
| `parameters.push_realization_noise_pct` | 0.1 |
| `parameters.push_procurement_waste_rate` | 0.15 |

#### Per-stage defaults (CONFIG.stages; UI can override base time, workers, transport, defect)

| Stage | base_process_time_min | transport_time_min | workers | rework_stage_id | defect_rate |
|-------|------------------------|--------------------|---------|------------------|-------------|
| S1 | 30.0 | 3.0 | 2 | S1 | 0.0 |
| S2 | 30.0 | 3.0 | 2 | S2 | 0.0 |
| S3 | 60.0 | 4.0 | 2 | S3 | 0.0 |
| S4 | 90.0 | 4.0 | 2 | S4 | 0.0 |
| S5 | 120.0 | 5.0 | 2 | S5 | 0.0 |

**Note:** Stage graph (S1→B→S2→C1/C2/C3→S3/S4/S5→E), buffer IDs, and model-specific BOMs/outputs are fixed in `CONFIG`; see `env.py` for full structure.

### KPI Tracking

- Per-model production tracking
- Unmet demand and overproduction detection
- Status indicators (Met/Unmet/Overproduced)
- Comprehensive cost accounting
- Buffer level time series

---

## 12) Usage Examples

### Running a Simulation

```python
from env import LegoLeanEnv, CONFIG

# Create environment
env = LegoLeanEnv(CONFIG, time_unit="min", seed=42)

# Run for 3 working weeks (default UI convention: 5 days/week × 8 hours/day)
# 3 weeks → 3 * 5 * 8 * 60 = 7,200 minutes
env.run_for(7200.0)

# Get KPIs
kpis = env.get_kpis()
print(f"Finished units: {kpis['finished_units']}")
print(f"Finished by model: {kpis['finished_goods_by_model']}")
print(f"Unmet demand (units): {kpis.get('unmet_demand_units', 0)}")
print(f"Overproduced (units): {kpis.get('overproduced_units', 0)}")
print(f"Revenue opportunity loss: {kpis.get('revenue_opportunity_loss', 0.0)}")
print(f"Overproduction waste cost: {kpis.get('overproduction_waste_cost', 0.0)}")
```

### Accessing Per-Model Data

```python
kpis = env.get_kpis()

# Per-model production
for model_id, qty in kpis['finished_goods_by_model'].items():
    print(f"{model_id}: {qty} units produced")

# Planned vs Realized
planned = kpis['planned_release_qty_by_model']
realized = kpis['demand_realized_by_model']
produced = kpis['finished_goods_by_model']

for model_id in planned.keys():
    print(f"{model_id}: Planned={planned[model_id]}, "
          f"Realized={realized.get(model_id, 0)}, "
          f"Produced={produced.get(model_id, 0)}")
```

---

## 13) Future Work / TODOs

The following improvements are planned:

### 1. Pull Mode Enhancements
- **Current status**: Pull mode (CONWIP + Kanban) and Order Release Configuration are implemented and used when “Enable Pull” is ticked
- **Possible enhancements**: Additional CONWIP/Kanban policies, more release-stage options, validation of Little’s Law under CONWIP

### 2. Adjust Time System
- **Current status**: Time is in minutes, but may need further adjustments
- **Potential improvements**:
  - Consider different time scales for different operations

### 3. Add Adjustable Forecast Probability
- **Current status**: Model probabilities are fixed in CONFIG
- **Enhancement needed**:
  - Add UI controls for adjusting model demand probabilities
  - Allow dynamic probability changes during simulation
  - Support time-varying demand patterns

### 4. KPI Testing
- **Current status**: KPIs are calculated but need validation
- **Testing needed**:
  - Validate KPI calculations against analytical models
  - Test Little's Law: L̄ = θ · W̄
  - Verify per-model KPI accuracy
  - Test unmet demand and overproduction calculations
  - Validate cost accounting accuracy

### Additional Potential Improvements

- **Advanced forecasting**: Add more sophisticated demand forecasting models
- **Optimization integration**: Connect simulation with optimization algorithms
- **Visualization enhancements**: Improve KPI charts and time series displays
- **Export functionality**: Add CSV/Excel export for KPI data

---

## 14) File Structure

```
Agile-in-Lean-Production/
├── README.md              # Project overview and quick start (links here)
├── env.py                 # Core DES engine, CONFIG, defect/rework logic
├── app.py                 # Streamlit UI (Push/Pull, Order Release, CONWIP/Kanban)
├── readme3.md             # This file — simulation model documentation
├── MULTI_MODEL_GUIDE.md   # Multi-model production guide
├── mathematical_model.md  # Mathematical model formulation
├── requirements.txt      # Python dependencies
└── baseline.txt          # Baseline model specifications
```

---

## 15) Getting Help

- **Multi-model configuration and usage**: [MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md) — model definitions, BOM examples, demand forecasting with multiple models, troubleshooting
- **Mathematical model**: [mathematical_model.md](mathematical_model.md) — objective, constraints, cost formulations
- **Code**: Inline comments in `env.py` and `app.py`

---

## 16) Validation Targets

- **Little's Law**: L̄ = θ · W̄ (should hold within simulation error)
- **Utilization**: ρ<sub>i</sub> < 0.85 for all stages (avoid bottlenecks)
- **Demand matching**: Minimize unmet demand and overproduction
- **Cost accuracy**: Material, labor, and inventory costs should match actual consumption

---

## Summary (key takeaways)

- **Dual mode**: Push (forecast-based release) or Pull (token-gated + CONWIP + Kanban); same DES engine and BOM/defect logic.
- **Time**: Simulation in **minutes**; processing time = base_time / √workers (constant); 24/7 production.
- **Multi-model**: Four product models (m01–m04); model-specific BOMs and outputs; FIFO order queue; see [MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md) for configuration.
- **Defect/rework**: Per-stage defect_rate and rework_stage_id; one rework per job, then scrap; rework bypasses token gating.
- **KPIs**: Throughput, WIP, lead time, service level, per-model planned/realized/produced, unmet/overproduced, cost and profit.
- **Configuration**: Single source in `env.py` CONFIG; UI overrides for mode, Order Release (Pull), CONWIP/Kanban, stage times, and buffer stocks. Target takt is not in CONFIG; the UI computes it in Pull mode (simulation time ÷ total orders) for the Flow Pacing display only.

---

**Last updated**: 2026-03-13  
**Version**: 2.1 (Push/Pull dual mode, defect/rework with `is_rework`, Order Release only in Pull)
