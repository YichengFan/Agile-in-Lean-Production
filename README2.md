# LEGO Lean Production — Simulation Model (Current Version)

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
- The system now supports **multi-model production** (4 different product models: m01-m04)
- **Push mode with demand forecasting** is the primary production mode
- Time units are in **minutes** (realistic production times)
- All configuration lives in `env.py`'s `CONFIG` dictionary
- Randomness is enabled by default; you can toggle Deterministic processing in the UI

---

## Mathematical Model

For the **mathematical model** (objective function, decision variables, constraints, cost formulations), please refer to [`mathematical_model.md`](mathematical_model.md).

This document focuses on the **simulation model** implementation details and current system capabilities.

---

# Simulation Model (Discrete-Event Simulation)

## 1) System Overview

The current system implements a **push-based production system with demand forecasting** that supports **multi-model production**. The system:

- Generates 3-week demand forecasts for multiple product models
- Allocates demand across models based on probability distribution
- Automatically queues production orders based on forecasts
- Tracks production performance per model
- Calculates KPIs including unmet demand and overproduction

### Production Modes

**Push Mode (Primary - Demand Forecasting):**
- Forecasts demand for a specified horizon (default: 3 weeks)
- Applies forecast noise and realization noise
- Automatically releases orders at simulation start
- Tracks planned vs. realized vs. produced quantities per model
- Calculates unmet demand and overproduction

**Pull Mode (Legacy - CONWIP + Kanban):**
- Token-gated order release
- CONWIP WIP cap control
- Kanban buffer caps
- Currently needs fixes (see Future Work)

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
- **Transport time**: δ<sub>i</sub> ≥ 0 (minutes)
- **Time distribution**: D<sub>i</sub> = (type, p<sub>i1</sub>, p<sub>i2</sub>, p<sub>i3</sub>)
  - Supported types: constant, normal, lognormal, triangular, uniform, exponential

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
  - Default: m01=60%, m02=20%, m03=10%, m04=10%
- **Forecast noise**: ε<sub>f</sub> ∈ [0, 1] (percentage applied to forecast generation)
- **Realization noise**: ε<sub>r</sub> ∈ [0, 1] (percentage applied to actual demand)
- **Procurement waste rate**: w ∈ [0, 1] (safety stock/waste percentage)

### Model-Specific Parameters

- **Model-specific BOMs**: Each stage can have different material requirements per model
  - `required_materials_by_model`: Dict[model_id, Dict[item_id, quantity]]
- **Model-specific outputs**: Each model can produce different finished goods
  - `output_buffers_by_model`: Dict[model_id, Dict[buffer_id, Dict[item_id, quantity]]]

### Other Parameters

- **Defect probability**: q<sub>i</sub> ∈ [0, 1] (default: 0.0 for all stages)
- **Buffer capacity**: cap<sub>b</sub> ∈ ℕ ∪ {∞}
- **Shift schedule**: Currently disabled (24/7 production)

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

### Order Release (Push Mode)

- Orders are automatically queued at simulation start based on forecast
- Each order is tagged with a `model_id`
- Orders are stored in `_stage_order_models[stage_id]` as a FIFO queue

### Try-Start at Stage i

A stage i can start only if all of the following hold:

1. **Idle condition**: Y<sub>i</sub>(t) = 0 (`stage.busy == False`)
2. **Release-stage token gating** (if i ∈ release_stage_ids):
   - `stage_orders[i] > 0` must hold
   - Starting consumes one token
   - Model_id is popped from the FIFO queue
3. **Material availability (BOM)**:
   - For release stages: model_id determines which BOM to use
   - For downstream stages: model_id is inferred from available materials
   - All required materials must be available in input buffers
4. **Kanban gating** (if enabled):
   - Output buffers must not exceed Kanban caps

If all checks pass:
- Pull BOM items from buffers (model-specific)
- Set Y<sub>i</sub>(t) = 1
- Calculate processing time: `base_time / √workers` + sample from distribution
- Schedule `complete` event at t + S<sub>i</sub>

### Completion at Stage i

On `complete(i)`:
- Stop team utilization timer U<sub>i</sub>
- **Defect handling** (if q<sub>i</sub> > 0):
  - Route to rework or scrap
- **Push outputs** (model-specific):
  - Use `output_buffers_by_model[model_id]` if available
  - Otherwise use default `output_buffers`
  - If downstream is blocked, retry later
  - If output goes to finished buffer E:
    - Increment `finished_goods_by_model[model_id]`
    - Decrement WIP

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
- **Utilization**: ρ<sub>i</sub> = U<sub>i</sub>(t) / t (per team)
- **Service level**: SL = N<sub>fin</sub>(t) / N<sub>target</sub>

### Demand & Production KPIs (Per Model)

- **Planned Release Quantity**: Forecast-based production plan per model
- **Realized Demand**: Actual customer demand per model (with noise)
- **Produced Quantity**: Actually finished units per model
- **Unmet Demand**: |realized - produced| (absolute difference)
- **Status**: Met / Unmet / Overproduced

### Quality KPIs

- **Defect rate per stage**: DR<sub>i</sub> = N<sub>i,def</sub> / (N<sub>i,def</sub> + N<sub>i,ok</sub>)
- **Total defect rate**: DR<sub>total</sub> = Σ N<sub>i,def</sub> / Σ (N<sub>i,def</sub> + N<sub>i,ok</sub>)

### Flow KPIs

- **Starvation counts**: Stage wants to start but lacks input materials
- **Blocking counts**: Stage finishes but downstream buffer is full
- **Kanban blocking counts**: Stage blocked by Kanban cap (if enabled)

### Cost & Profit KPIs

- **Revenue**: `unit_price` × finished (or capped by demand if provided)
- **Material cost**: `unit_material_cost` × consumed quantity
- **Labor cost**: Σ (team busy time × team size × labor_rate)
- **Inventory holding cost**: Σ (∫inventory dt × holding_rate) (Buffer A excluded)
- **Profit**: Revenue − (Material + Labor + Inventory + Other)
- **Opportunity cost**: `margin_per_unit` × unmet_demand

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
        "push_weekly_demand_mean": 30,
        "push_forecast_noise_pct": 0.1,
        "push_realization_noise_pct": 0.05,
        "push_procurement_waste_rate": 0.05,
        "model_demand_probabilities": {
            "m01": 0.6, "m02": 0.2, "m03": 0.1, "m04": 0.1
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

The Streamlit UI allows interactive configuration of:
- Production mode (Push/Pull)
- Demand forecasting parameters
- Stage processing times
- Buffer initial stocks
- Team sizes
- Time distributions

---

## 9) Simulation Implementation Mapping

| Simulation Element | `env.py` Implementation |
|-------------------|------------------------|
| τ<sub>i</sub>, D<sub>i</sub> | `base_process_time_min`, `time_distribution` |
| w<sub>i</sub> | `workers` |
| Effective time | `base_time / √workers` (deterministic) |
| δ<sub>i</sub> | `transport_time_min` |
| q<sub>i</sub> | `defect_rate` (default: 0.0) |
| Model-specific BOM | `required_materials_by_model[model_id]` |
| Model-specific outputs | `output_buffers_by_model[model_id]` |
| Forecast parameters | `push_*` parameters in `CONFIG` |
| Model probabilities | `model_demand_probabilities` |
| Order queue | `_stage_order_models[stage_id]` (FIFO) |
| WIP L(t) | `self.current_wip` |
| Lead time tracking | `self._release_times` (deque) |
| Finished per model | `self.finished_goods_by_model` |
| Planned per model | `self.planned_release_qty` |
| Realized per model | `self.demand_realized` |

---

## 10) Simulation Assumptions

- **Discrete-event simulation**: Events processed in chronological order using a priority queue
- **Push strategy (primary)**: Orders released automatically based on demand forecast
- **Time unit**: Minutes (realistic production times)
- **Worker efficiency**: Deterministic square root function (base_time / √workers)
- **24/7 production**: Shift schedules disabled for continuous operation
- **Model-specific BOMs**: Each model can have different material requirements
- **FIFO order processing**: Orders processed in first-in-first-out order per model
- **Material procurement**: Materials added to Buffer A at start based on forecast + waste
- **No missing bricks**: Removed as a source of variability (time distribution handles variability)

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

### Realistic Time System

- Time in minutes (not seconds)
- Deterministic worker efficiency
- Configurable time distributions
- Transport times per stage

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

# Run for 3 weeks (30240 minutes)
env.run_for(30240.0)

# Get KPIs
kpis = env.get_kpis()
print(f"Finished units: {kpis['finished_units']}")
print(f"Finished by model: {kpis['finished_goods_by_model']}")
print(f"Unmet demand: {kpis.get('unmet_demand_total', 0)}")
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

### 1. Fix Pull Production System
- **Current status**: Pull mode (CONWIP + Kanban) exists but needs fixes
- **Issues to address**:
  - Ensure proper token gating
  - Fix CONWIP WIP cap enforcement
  - Verify Kanban buffer cap logic
  - Test closed-loop CONWIP behavior

### 2. Adjust Time System
- **Current status**: Time is in minutes, but may need further adjustments
- **Potential improvements**:
  - Consider different time scales for different operations
  - Add time-based shift scheduling (currently disabled)
  - Optimize time distribution parameters

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

- **Shift scheduling**: Re-enable and test shift-based production
- **Advanced forecasting**: Add more sophisticated demand forecasting models
- **Optimization integration**: Connect simulation with optimization algorithms
- **Visualization enhancements**: Improve KPI charts and time series displays
- **Export functionality**: Add CSV/Excel export for KPI data

---

## 14) File Structure

```
Agile-in-Lean-Production/
├── env.py                 # Core simulation engine and CONFIG
├── app.py                 # Streamlit UI for simulation control
├── README.md              # Original documentation (outdated)
├── README2.md             # This file - current documentation
├── MULTI_MODEL_GUIDE.md   # Detailed multi-model production guide
├── mathematical_model.md # Mathematical model formulation
├── requirements.txt       # Python dependencies
└── baseline.txt          # Baseline model specifications
```

---

## 15) Getting Help

- **Multi-model production**: See [`MULTI_MODEL_GUIDE.md`](MULTI_MODEL_GUIDE.md) for detailed examples
- **Mathematical model**: See [`mathematical_model.md`](mathematical_model.md) for optimization formulation
- **Code documentation**: See inline comments in `env.py` and `app.py`

---

## 16) Validation Targets

- **Little's Law**: L̄ = θ · W̄ (should hold within simulation error)
- **Utilization**: ρ<sub>i</sub> < 0.85 for all stages (avoid bottlenecks)
- **Demand matching**: Minimize unmet demand and overproduction
- **Cost accuracy**: Material, labor, and inventory costs should match actual consumption

---

**Last Updated**: 2026-01-XX  
**Version**: 2.0 (Multi-Model with Demand Forecasting)
