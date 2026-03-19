# Multi-Model Support Guide

**Purpose:** This guide explains how to **configure and use multi-model production** in the LEGO Lean Production simulation: four product variants (m01–m04), model-specific BOMs, demand forecasting per model, and Push/Pull usage.

**Relation to [simulation_overview.md](simulation_overview.md):** simulation_overview is the simulation specification (event logic, parameters, KPIs). This guide **extends** it with multi-model–specific configuration, workflow, and troubleshooting. For event logic, parameter tables, and implementation mapping, use readme3.

---

## Table of Contents

- [Overview](#overview)
- [Model Definitions](#model-definitions)
- [Key Features](#key-features)
- [Configuration](#configuration) (model definitions, materials per stage, default model)
- [Usage](#usage) (Push with forecasting, manual release, mixed production)
- [How It Works](#how-it-works) (order release, material/output selection, FIFO, demand forecasting)
- [Partial Model Support](#partial-model-support)
- [Current Implementation Status](#current-implementation-status)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Demand Forecasting and Multi-Model Integration](#demand-forecasting-and-multi-model-production-integration)
- [Quick Reference](#quick-reference-demand-forecasting--multi-model-production)
- [Future Enhancements](#future-enhancements)

---

## Overview

The system supports **4 different model types** (m01, m02, m03, m04), each with potentially different material requirements at different stages. This allows you to simulate production lines that produce multiple product variants.

## Model Definitions

The system includes four pre-configured models:

| Model ID | Model Name | Description |
|----------|-----------|-------------|
| m01 | Sly_Slider | Sly Slider model configuration |
| m02 | Gliderlinski | Gliderlinski model configuration |
| m03 | Icky_Ice_Glider | Icky Ice Glider model configuration |
| m04 | Icomat_2000X | Icomat 2000X model configuration |

## Key Features

1. **Model-specific material requirements**: Each stage can have different material requirements for different models
2. **Model-specific outputs**: Different models can produce different finished goods
3. **Model tracking**: Each order/job is tagged with a model_id that follows it through the production process
4. **Backward compatibility**: If no model_id is specified, the system uses default materials (legacy behavior)
5. **FIFO queueing**: Orders are processed in First-In-First-Out order based on when they were enqueued

**Quick orientation:** In **Push** mode, demand is forecast and orders are released automatically at simulation start; in **Pull** mode, the user sets Order Release (quantity per model) and CONWIP/Kanban in the UI. Both modes use the same DES engine and model-specific BOMs. Full parameter list and defaults → [readme3.md](readme3.md) (Fixed Configuration Parameters and §8 Configuration).

## Configuration

### 1. Model Definitions (Already Configured)

Models are defined in the `CONFIG` dictionary under the `"models"` section:

```python
"models": {
    "m01": {
        "name": "Sly_Slider",
        "description": "Sly Slider model configuration"
    },
    "m02": {
        "name": "Gliderlinski",
        "description": "Gliderlinski model configuration"
    },
    "m03": {
        "name": "Icky_Ice_Glider",
        "description": "Icky Ice Glider model configuration"
    },
    "m04": {
        "name": "Icomat_2000X",
        "description": "Icomat 2000X model configuration"
    }
}
```

### 2. Configure Model-Specific Materials per Stage

For each stage where you want model-specific behavior, configure:

- `required_materials_by_model`: Dictionary mapping model_id to material requirements
- `output_buffers_by_model`: Dictionary mapping model_id to output specifications

#### Example: S2 (Set Sorting / Kit Build)

S2 routes to C1, C2, C3 with model-specific kit items. See `env.py` CONFIG for the full BOM; structure only:

```python
{
    "stage_id": "S2",
    "name": "Set Sorting (Kit Build)",
    # ... other fields (input_buffers ["B"], team_id, base_process_time_min, etc.) ...
    
    # Model-specific outputs (one kit per output buffer per model)
    "output_buffers_by_model": {
        "m01": {"C1": {"buna01": 1}, "C2": {"bunc01": 1}, "C3": {"bunf01": 1}},
        "m02": {"C1": {"buna02": 1}, "C2": {"bunc02": 1}, "C3": {"bunf02": 1}},
        "m03": {"C1": {"buna03": 1}, "C2": {"bunc03": 1}, "C3": {"bunf03": 1}},
        "m04": {"C1": {"buna04": 1}, "C2": {"bunc04": 1}, "C3": {"bunf04": 1}}
    }
}
```

#### Example: S3 (Axis Subassembly)

S3 pulls from C1 (model-specific kit) and outputs to D1. See `env.py` CONFIG for full BOM:

```python
{
    "stage_id": "S3",
    "name": "Axis Subassembly",
    "input_buffers": ["C1"],
    "required_materials_by_model": {
        "m01": {"buna01": 1},  # Sly_Slider
        "m02": {"buna02": 1},  # Gliderlinski
        "m03": {"buna03": 1},  # Icky_Ice_Glider
        "m04": {"buna04": 1}   # Icomat_2000X
    },
    "output_buffers": {"D1": {"saa01": 1}},
    "output_buffers_by_model": {
        "m01": {"D1": {"saa01": 1}},
        "m02": {"D1": {"saa02": 1}},
        "m03": {"D1": {"saa02": 1}},
        "m04": {"D1": {"saa01": 1}}
    }
}
```

#### Example: S5 (Final Assembly)

S5 pulls from C3, D1, D2 (model-specific items) and produces one finished good per model. See `env.py` CONFIG for full BOM:

```python
{
    "stage_id": "S5",
    "name": "Final Assembly",
    "input_buffers": ["C3", "D1", "D2"],
    "required_materials_by_model": {
        "m01": {"bunf01": 1, "saa01": 1, "sac01": 1},  # Sly_Slider
        "m02": {"bunf02": 1, "saa02": 1, "sac01": 1},  # Gliderlinski
        "m03": {"bunf03": 1, "saa02": 1, "sac02": 1},  # Icky_Ice_Glider
        "m04": {"bunf04": 1, "saa01": 1, "sac02": 1}  # Icomat_2000X
    },
    "output_buffers": {"E": {"fg01": 1}},
    "output_buffers_by_model": {
        "m01": {"E": {"fg01": 1}},
        "m02": {"E": {"fg02": 1}},
        "m03": {"E": {"fg03": 1}},
        "m04": {"E": {"fg04": 1}}
    }
}
```

### 3. Set Default Model

In `parameters`, specify the default model:

```python
"parameters": {
    "default_model_id": "m01",  # Default model if not specified
    # ... other parameters
}
```

## Usage

### Push Mode with Demand Forecasting (Recommended)

When using push mode with demand forecasting, orders are automatically queued based on forecasts:

```python
from env import LegoLeanEnv, CONFIG

# Configure push mode (forecast automatically triggers production)
CONFIG["parameters"]["push_demand_enabled"] = True
CONFIG["parameters"]["push_weekly_demand_mean"] = 25  # 25 units/week × 3 weeks (CONFIG default)
CONFIG["parameters"]["model_demand_probabilities"] = {
    "m01": 0.60,  # Sly_Slider 60%
    "m03": 0.20,  # Icky_Ice_Glider 20%
    "m02": 0.10,  # Gliderlinski 10%
    "m04": 0.10   # Icomat_2000X 10%
}

env = LegoLeanEnv(CONFIG, time_unit="min", seed=42)

# Forecast is generated automatically, orders are queued immediately
# No need to manually call enqueue_orders() - it's done automatically!

# Run simulation
env.run_for(30240.0)  # 3 weeks in minutes

# Check results
kpis = env.get_kpis()
print(f"Planned: {kpis['planned_release_qty']}")
print(f"Realized: {kpis['demand_realized_total']}")
print(f"Produced: {kpis['finished_units']}")
print(f"Unmet Demand: {max(0, kpis['demand_realized_total'] - kpis['finished_units'])}")

# Per-model breakdown
for model_id in ["m01", "m02", "m03", "m04"]:
    planned = kpis["planned_release_qty_by_model"].get(model_id, 0)
    realized = kpis["demand_realized_by_model"].get(model_id, 0)
    produced = kpis["finished_goods_by_model"].get(model_id, 0)
    unmet = max(0, realized - produced)
    print(f"{model_id}: Planned={planned}, Realized={realized}, Produced={produced}, Unmet={unmet}")
```

### Manual Order Release (Pull Mode or Manual Push)

When releasing orders manually (without demand forecasting), specify the model type:

```python
from env import LegoLeanEnv, CONFIG

env = LegoLeanEnv(CONFIG, time_unit="min", seed=42)

# Release 10 orders of Model 1 (Sly_Slider)
env.enqueue_orders(qty=10, model_id="m01")

# Release 5 orders of Model 2 (Gliderlinski)
env.enqueue_orders(qty=5, model_id="m02")

# Release orders of different models (FIFO queueing)
env.enqueue_orders(qty=1, model_id="m01")  # First order
env.enqueue_orders(qty=2, model_id="m02")  # Second and third orders
# Processing order: m01 → m02 → m02

# Release without specifying model (uses default_model_id)
env.enqueue_orders(qty=3)  # Will use "m01" if default_model_id="m01"

# CONWIP replenishment (internal use): is_replenishment=True so these don't count toward demand for service level
# env.enqueue_orders(qty=1, model_id=..., is_replenishment=True)  # called by env when a unit finishes in pull mode
```

### Mixed Model Production

To produce multiple models in one simulation run:

```python
# Release multiple models with different quantities
env.enqueue_orders(qty=1, model_id="m01")   # 1 Sly_Slider
env.enqueue_orders(qty=2, model_id="m02")   # 2 Gliderlinski
env.enqueue_orders(qty=1, model_id="m03")   # 1 Icky_Ice_Glider
env.enqueue_orders(qty=1, model_id="m04")   # 1 Icomat_2000X

# Run simulation
env.run_for(3600.0)  # Run for 1 hour

# Results: Will produce exactly 1 m01, 2 m02, 1 m03, 1 m04
# Processing order: m01 → m02 → m02 → m03 → m04 (FIFO)
```

## How It Works

### Order Release and Tracking

1. **Order Release**: When `enqueue_orders(qty, model_id)` is called:
   - The model_id is stored in a FIFO queue (`_stage_order_models`) for each release stage
   - For example: `enqueue_orders(1, "m01")` followed by `enqueue_orders(2, "m02")` creates queue: `["m01", "m02", "m02"]`
   - **In push mode with auto-release**: This happens automatically in `_init_push_demand_plan()` for each model's `planned_release_qty`

2. **Job Start**: When a stage starts processing:
   - **For release stages (e.g., S1)**: model_id is retrieved from the queue using `.pop(0)` (FIFO)
   - **For downstream stages**: model_id is passed through events from upstream stages
   - If model_id is not available, the default model is used

3. **Material Selection**: 
   - The system first checks `required_materials_by_model[model_id]`
   - If found, uses those materials
   - Otherwise, falls back to `required_materials` (default/legacy)

4. **Output Selection**:
   - Similar to materials, checks `output_buffers_by_model[model_id]` first
   - Falls back to `output_buffers` if not found

5. **Model Tracking**: 
   - Each job gets a unique `job_id`
   - The `job_id` is mapped to `model_id` in `_job_model_map`
   - This allows model_id to be retrieved even if not explicitly passed in events

### Demand Forecasting Integration Flow

When `push_demand_enabled=True`:

1. **Initialization**: `_init_push_demand_plan()` is called during `__init__`
2. **Forecast Generation**: 
   - Total forecast calculated: `weekly_mean × 3 × (1 ± forecast_noise)`
   - Allocated across models by probability: `model_forecast = total × probability`
   - Realized demand calculated: `forecast × (1 ± realization_noise)`
3. **Automatic Order Queuing**: For each model with `planned_release_qty > 0` (always enabled):
   ```python
   self.enqueue_orders(qty=planned_qty, model_id=model_id)
   ```
4. **Production Start**: Orders enter FIFO queue, production begins immediately
5. **Tracking**: System tracks Planned, Realized, and Produced per model, calculates Unmet Demand

### FIFO Queueing

The system uses a simple FIFO (First-In-First-Out) queue:

- Orders are enqueued using `.extend([model_id] * qty)` which maintains order
- Orders are dequeued using `.pop(0)` which removes the first element
- **Example**: If you enqueue `1x m01`, then `2x m02`, the processing order will be: `m01 → m02 → m02`

## Partial Model Support

You don't need to define materials for all models at all stages. The system will:

- Use model-specific materials if defined for that model at that stage
- Fall back to default materials if not defined
- This allows you to have some stages with model-specific behavior and others with shared behavior

**Example**: 
- If S1 (Type Sorting) uses the same materials for all models, you only need to define `required_materials`
- But if S5 (Final Assembly) has different requirements per model, define `required_materials_by_model`

## Current Implementation Status

### ✅ Implemented

- Model definitions (m01-m04 with proper names)
- Model-specific material configuration structures in S2, S3, S4, S5 (see `env.py` CONFIG for exact item IDs)
- Model tracking through job lifecycle
- FIFO queueing for orders
- Programmatic API support (`enqueue_orders(qty, model_id=..., is_replenishment=...)`)
- **Processing time**: Constant per stage; effective time = `base_process_time_min / sqrt(workers)` (no random distributions)
- **Defect/rework**: Per-stage `defect_rate` and `rework_stage_id`; one rework allowed, then scrap (`is_rework` in events)
- **Time unit**: Simulation time is in **minutes** (CONFIG and `run_for` use minutes)
- **Pull mode**: In CONWIP, replenishment orders are enqueued with `is_replenishment=True` so they do not count toward demand; service level uses `initial_orders_released` as demand total

### 📝 UI Integration (Optional)

The current Streamlit UI (`app.py`) uses a simple single-model approach. To support multi-model selection in the UI, you can:

1. Add quantity inputs for each model in the sidebar
2. Collect `model_quantities` dictionary from UI
3. Loop through and call `env.enqueue_orders(qty=qty, model_id=model_id)` for each

Example UI integration:

```python
# In app.py sidebar section
model_definitions = cfg.get("models", {})
model_quantities = {}

for model_id, model_info in model_definitions.items():
    qty = st.number_input(
        f"{model_info['name']} ({model_id})",
        min_value=0,
        value=0,
        step=1,
        key=f"model_qty_{model_id}"
    )
    if qty > 0:
        model_quantities[model_id] = qty

# In simulation run section
for model_id, qty in model_quantities.items():
    if qty > 0:
        env.enqueue_orders(qty=int(qty), model_id=model_id)
```

## Best Practices

1. **Start Simple**: Begin with one model (m01) and default materials. Add model-specific configurations gradually.

2. **Consistent Naming**: Use consistent model IDs (m01, m02, m03, m04) across all stages.

3. **Document Differences**: In your model definitions and comments, clearly document what makes each model different.

4. **Test Incrementally**: Test each model separately before mixing them in production runs.

5. **Buffer Considerations**: Different models may produce different items. Make sure your buffers can handle all item types that will be produced.

6. **Material Planning**: When defining model-specific materials, ensure that:
   - Required materials are available in input buffers
   - Material quantities make sense for the production process
   - Different models don't create material conflicts

## Troubleshooting

### Issue: Model-specific materials not being used
- **Check**: Ensure `required_materials_by_model` is defined in the stage config
- **Check**: Verify the model_id matches exactly (case-sensitive: "m01" not "M01")
- **Check**: Ensure model_id is being passed correctly through `enqueue_orders()`

### Issue: Default materials always used
- **Check**: Verify `default_model_id` is set in parameters
- **Check**: Ensure models are defined in the `models` section of CONFIG
- **Check**: Verify that `required_materials_by_model[model_id]` exists for your model

### Issue: Model_id not propagating to downstream stages
- **Check**: The system passes model_id through events. If a stage doesn't receive model_id, it uses the default.
- **Note**: In complex routing scenarios, you may need to track model_id per buffer item (advanced feature not currently implemented).

### Issue: Wrong processing order (not FIFO)
- **Check**: Verify that `enqueue_orders()` is being called in the desired order
- **Check**: Ensure you're not using parallel processing that might affect order
- **Note**: FIFO is maintained per release stage. If you have multiple release stages, each maintains its own queue.

### Issue: Demand forecast not triggering production
- **Check**: Ensure `push_demand_enabled=True` in parameters
- **Check**: Verify `model_demand_probabilities` is configured for all models
- **Check**: Check simulation logs for `[PUSH] Queued X order(s)` messages
- **Note**: In push mode, forecast automatically triggers production (no manual intervention needed)

### Issue: Unmet demand in push mode
- **Expected behavior**: In push mode, production is planned based on forecast, not realized demand
- **If realized > forecast**: Unmet demand occurs (push mode limitation)
- **If realized < forecast**: Overproduction occurs
- **Solution**: This is by design - push mode plans production in advance. Use pull mode for demand-driven production.

### Issue: All models getting same quantity
- **Check**: Verify `model_demand_probabilities` sums to 1.0
- **Check**: Ensure probabilities are correctly configured (e.g., m01=0.60, not 60)
- **Check**: Verify forecast allocation logic in `_init_push_demand_plan()`

## Example: Complete Multi-Model Configuration

The full CONFIG (stages, buffers, parameters) is in **env.py**. For the complete multi-model setup, see that file; it includes:
- 4 model definitions (m01-m04: Sly_Slider, Gliderlinski, Icky_Ice_Glider, Icomat_2000X)
- Model-specific materials at S2, S3, S4, S5
- Model-specific outputs producing different finished goods (fg01-fg04)
- Default model set to m01

## Demand Forecasting and Multi-Model Production Integration

### Overview

The system integrates **demand forecasting** with **multi-model production** to create a realistic push-mode production planning system. When enabled, the system:

1. **Generates per-model demand forecasts** based on probability distributions
2. **Automatically queues production orders** for each model based on forecasts
3. **Tracks realized demand** separately for each model to measure forecast accuracy
4. **Produces multiple models** simultaneously through the shared production line

This creates a realistic scenario where:
- Production is planned based on forecasts (which may be inaccurate)
- Multiple product variants are produced in the same production line
- Unmet demand can occur if realized demand exceeds forecast (push mode limitation)
- Overproduction can occur if realized demand is lower than forecast

### Complete Workflow: From Forecast to Production

Here's the step-by-step process of how demand forecasting triggers multi-model production:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SIMULATION INITIALIZATION                                    │
│    - push_demand_enabled = True                                 │
│    - Auto-release always enabled in push mode                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. FORECAST GENERATION (_init_push_demand_plan)                 │
│    a) Calculate total 3-week forecast:                        │
│       total_forecast = weekly_mean × 3 × (1 ± forecast_noise)  │
│    b) Allocate across models by probability:                    │
│       m01: 60% → 54 (example; with weekly_mean=25 total ~75)   │
│       m03: 20% → 18  m02: 10% → 9  m04: 10% → 9                │
│    c) Generate realized demand (with noise):                    │
│       realized = forecast × (1 ± realization_noise)            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. AUTOMATIC ORDER QUEUING (always enabled in push mode)       │
│    For each model:                                              │
│      enqueue_orders(qty=planned_qty, model_id=model_id)        │
│    Example:                                                      │
│      enqueue_orders(54, "m01")  # Sly_Slider                   │
│      enqueue_orders(18, "m03")  # Icky_Ice_Glider              │
│      enqueue_orders(9, "m02")   # Gliderlinski                 │
│      enqueue_orders(9, "m04")   # Icomat_2000X                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. FIFO QUEUE POPULATION                                        │
│    Each release stage (e.g., S1) gets a queue:                  │
│      _stage_order_models["S1"] = [                             │
│        "m01", "m01", ..., (54 times),                           │
│        "m03", "m03", ..., (18 times),                          │
│        "m02", "m02", ..., (9 times),                            │
│        "m04", "m04", ..., (9 times)                             │
│      ]                                                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. PRODUCTION EXECUTION                                         │
│    As stages process orders (FIFO):                             │
│      - S1 pops model_id from queue → uses model-specific BOM   │
│      - S2 receives model_id → uses model-specific materials    │
│      - S3, S4, S5 continue with model_id tracking              │
│      - S5 produces model-specific finished goods:              │
│        m01 → fg01, m02 → fg02, m03 → fg03, m04 → fg04         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. KPI TRACKING                                                 │
│    - planned_release_qty_by_model: Planned production per model │
│    - demand_realized_by_model: Actual demand per model          │
│    - finished_goods_by_model: Actual production per model       │
│    - unmet_demand: realized - produced (if realized > produced)│
└─────────────────────────────────────────────────────────────────┘
```

### Key Integration Points

1. **Forecast → Order Queue**: In push mode, the system automatically calls `enqueue_orders()` for each model's `planned_release_qty` at simulation start. This directly connects forecast to production queue (always enabled in push mode).

2. **Model-Specific Queuing**: Each `enqueue_orders(qty, model_id)` call adds `model_id` entries to the FIFO queue, maintaining the order of release.

3. **Production Flow**: The `model_id` flows through the production system:
   - **S1 (Type Sorting)**: Retrieves `model_id` from queue, uses `required_materials_by_model[model_id]`
   - **S2-S5**: Receive `model_id` from upstream events, use model-specific materials and produce model-specific outputs

4. **Demand Tracking**: The system tracks three key metrics:
   - **Planned production** (`planned_release_qty`): Based on forecast (what we plan to make)
   - **Realized demand** (`realized_demand_total`): Actual demand with noise (what customers want)
   - **Produced goods** (`finished_goods_by_model`): What we actually made
   - **Unmet demand**: `realized - produced` (if realized > produced)

## Per-Model Demand Forecasting (Phase 2)

### Overview

The system now supports **per-model demand forecasting** for push-mode production planning. Instead of generating a single aggregate forecast, the system generates separate forecasts for each model and allocates demand based on a probability distribution.

### Model Probability Distribution

By default, demand is allocated across models as follows:

| Model ID | Model Name | Probability |
|----------|-----------|-------------|
| m01 | Sly_Slider | 60% |
| m03 | Icky_Ice_Glider | 20% |
| m02 | Gliderlinski | 10% |
| m04 | Icomat_2000X | 10% |

### Configuration

The model probability distribution is configured in `parameters`:

```python
"parameters": {
    "model_demand_probabilities": {
        "m01": 0.60,  # Sly_Slider 60%
        "m03": 0.20,  # Icky_Ice_Glider 20%
        "m02": 0.10,  # Gliderlinski 10%
        "m04": 0.10   # Icomat_2000X 10%
    },
    # ... other parameters
}
```

### Forecast Generation

When `push_demand_enabled=True`, the system:

1. **Generates total forecast for 3 weeks** (not per-week, but total over 3 weeks)
2. **Allocates total demand** across models based on probability distribution
3. **Generates per-model forecasts** with forecast noise applied
4. **Generates per-model realizations** with realization noise applied
5. **Automatically queues production orders** if `push_auto_release=True`

**Example:**
- Total 3-week forecast: 90 units (with ±10% forecast noise)
- Allocation: m01=54 (60%), m03=18 (20%), m02=9 (10%), m04=9 (10%)
- Each model gets its own forecast and realization values
- **Production orders are immediately queued**: 54×m01, 18×m03, 9×m02, 9×m04
- Realized demand may differ: e.g., m01 realized=58 (forecast was 54) → unmet demand = 4

### Data Structure

Forecast data is stored per-model:

```python
# Per-model structure
self.demand_forecast = {
    "m01": [54],  # Sly_Slider forecast for 3 weeks
    "m02": [9],   # Gliderlinski forecast
    "m03": [18],  # Icky_Ice_Glider forecast
    "m04": [9]    # Icomat_2000X forecast
}

self.planned_release_qty = {
    "m01": 54,
    "m02": 9,
    "m03": 18,
    "m04": 9
}
```

### KPI Reporting

KPIs now include per-model breakdowns showing Planned, Realized, Produced, and Unmet Demand:

```python
kpis = env.get_kpis()
# Aggregate totals
kpis["planned_release_qty"]  # Total planned production (sum of all models)
kpis["demand_realized_total"]  # Total realized demand (sum of all models)
kpis["finished_units"]  # Total produced (sum of all models)
# Unmet demand = demand_realized_total - finished_units

# Per-model breakdowns
kpis["planned_release_qty_by_model"]  # Dict[str, int] - What we planned to produce
kpis["demand_realized_by_model"]  # Dict[str, int] - What customers actually want
kpis["finished_goods_by_model"]  # Dict[str, int] - What we actually produced
# Per-model unmet = demand_realized_by_model[model_id] - finished_goods_by_model[model_id]
kpis["service_level"]  # finished / demand_total (in pull mode demand_total = initial_orders_released)
kpis["overproduced_units"]  # Finished units in excess of demand (push/pull)
```

## Automatic Order Release in Push Mode

**When `push_demand_enabled=True`**, the system **automatically releases orders** based on the forecast:

- All forecasted orders are queued **at simulation start (t=0)**
- Orders are immediately available for production
- This simulates a "push" system where production is planned in advance
- All models' orders are queued simultaneously, maintaining FIFO order
- **No manual intervention needed** - forecast automatically triggers production

**Example:**
```python
# At t=0, immediately after forecast generation (automatic):
enqueue_orders(qty=54, model_id="m01")  # All 54 Sly_Slider orders
enqueue_orders(qty=18, model_id="m03")  # All 18 Icky_Ice_Glider orders
enqueue_orders(qty=9, model_id="m02")   # All 9 Gliderlinski orders
enqueue_orders(qty=9, model_id="m04")   # All 9 Icomat_2000X orders

# Production queue: [m01×54, m03×18, m02×9, m04×9]
# Processing order: FIFO (first m01, then m03, then m02, then m04)
```

**Note**: When `push_demand_enabled=True`, the code forces auto-release on (the `push_auto_release` parameter still exists in CONFIG but is ignored in push mode). Push mode means production is driven by forecast, not by actual demand.

## Changes Summary

### Phase 2: Per-Model Demand Forecasting

**Added:**
- `model_demand_probabilities` configuration parameter
- Per-model forecast generation (`demand_forecast: Dict[str, List[int]]`)
- Per-model realization tracking (`demand_realized: Dict[str, List[int]]`)
- Per-model planned release quantities (`planned_release_qty: Dict[str, int]`)
- Per-model KPI reporting

**Changed:**
- `_init_push_demand_plan()` now generates per-model forecasts for 3 weeks total
- Forecast allocation based on probability distribution instead of equal distribution
- KPI calculations updated to handle per-model data structures

### Phase 3: Demand Forecast → Production Integration

**Added:**
- Automatic order queuing in `_init_push_demand_plan()` (always enabled in push mode)
- Direct connection between forecast generation and production queue
- Per-model order queuing with `model_id` parameter
- `finished_goods_by_model` tracking for per-model production KPIs

**Changed:**
- `_init_push_demand_plan()` now directly calls `enqueue_orders()` for each model
- Orders are queued immediately at simulation start (immediate release mode)
- Each model's forecast automatically triggers production orders
- When `push_demand_enabled=True`, auto-release is always on (`push_auto_release` in CONFIG is overridden)
- Removed `demand_forecast_total` KPI (equals `planned_release_qty`)
- Replaced `started_units` with `planned_release_qty_by_model` in KPIs
- KPI structure: Planned, Realized, Produced, Unmet Demand, service_level, overproduced_units

**How It Works:**
1. Forecast generation creates `planned_release_qty` for each model
2. System automatically calls `enqueue_orders(qty, model_id)` for each model (always enabled)
3. Orders enter FIFO queue with model_id preserved
4. Production starts processing orders in FIFO order, using model-specific materials
5. System tracks `finished_goods_by_model` to measure actual production per model

## Quick Reference: Demand Forecasting + Multi-Model Production

### Configuration Checklist

```python
CONFIG["parameters"] = {
    # Enable demand forecasting
    # Note: When push_demand_enabled=True, forecast automatically triggers production (auto-release always enabled)
    "push_demand_enabled": True,
    
    # Forecast parameters
    "push_demand_horizon_weeks": 3,
    "push_weekly_demand_mean": 25,  # Total weekly demand (CONFIG default)
    "push_forecast_noise_pct": 0.2,  # ±20% forecast uncertainty (CONFIG default)
    "push_realization_noise_pct": 0.1,  # ±10% demand variation (CONFIG default)
    
    # Model probability distribution
    "model_demand_probabilities": {
        "m01": 0.60,  # Sly_Slider 60%
        "m03": 0.20,  # Icky_Ice_Glider 20%
        "m02": 0.10,  # Gliderlinski 10%
        "m04": 0.10   # Icomat_2000X 10%
    },
    
    # Default model (fallback)
    "default_model_id": "m01"
}
```

### Key Data Structures

```python
# Per-model forecast data
env.demand_forecast = {
    "m01": [54],  # Forecast for 3 weeks
    "m02": [9],
    "m03": [18],
    "m04": [9]
}

# Planned production (based on forecast)
env.planned_release_qty = {
    "m01": 54,  # Orders queued for production
    "m02": 9,
    "m03": 18,
    "m04": 9
}

# Realized demand (actual demand with noise)
env.realized_demand_total = {
    "m01": 58,  # May differ from forecast
    "m02": 8,
    "m03": 19,
    "m04": 10
}

# Finished goods (actual production per model)
env.finished_goods_by_model = {
    "m01": 54,  # Actually produced
    "m02": 9,
    "m03": 18,
    "m04": 9
}
```

### Production Flow Summary

1. **Forecast** → Generates per-model demand predictions (planned_release_qty)
2. **Queue** → Automatically queues orders for each model (`enqueue_orders`) - always enabled in push mode
3. **Produce** → FIFO processing with model-specific materials
4. **Track** → Monitors Planned vs Realized vs Produced per model
5. **Measure** → KPIs show:
   - **Planned**: What we planned to produce (based on forecast)
   - **Realized**: What customers actually want (forecast + realization noise)
   - **Produced**: What we actually made (finished_goods_by_model)
   - **Unmet Demand**: Realized - Produced (if realized > produced)

## Future Enhancements

Potential improvements for future versions:

1. **Priority-based queueing**: Order models by priority instead of FIFO
2. **Round-robin scheduling**: Alternate between models
3. **Model-specific processing times**: Different models might take different times to process
4. **Model-specific defect rates**: Different models might have different quality characteristics
5. **Buffer-level model tracking**: Track which model produced each item in buffers (for advanced routing)
6. **UI enhancement**: Built-in multi-model selection interface in Streamlit app
7. **Flexible release strategies**: Continuous vs batch production scheduling
8. **Dynamic demand updates**: Update forecasts during simulation based on realized demand
9. **Model-specific forecast accuracy**: Different models may have different forecast uncertainty

---

**See also:** [simulation_overview.md](simulation_overview.md) for event logic, KPIs, and parameter reference.
