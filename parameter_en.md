# LEGO Lean Production Simulation - Parameters and Process Documentation

## Table of Contents
1. [Overall Production Flow](#overall-production-flow)
2. [Stage Parameters](#stage-parameters)
3. [Buffer Parameters](#buffer-parameters)
4. [Team Parameters](#team-parameters)
5. [Key Variables](#key-variables)
6. [Event Types and Logic](#event-types-and-logic)
7. [KPI Metrics](#kpi-metrics)
8. [Global Parameters](#global-parameters)

---

## Overall Production Flow

### Flow Diagram
```
S1 (Type Sorting) 
  ↓
B (Warehouse B)
  ↓
S2 (Set Sorting) 
  ↓ (probabilistic routing)
  ├─→ C1 (40%) → S3 (Axis Assembly) → D1 ─┐
  ├─→ C2 (40%) → S4 (Chassis Assembly) → D2 ┼→ S5 (Final Assembly) → E
  └─→ C3 (20%) ────────────────────────────┘
```

### Process Description
1. **S1 (Type Sorting)**: Source stage with no input buffers, produces directly and pushes to B
2. **B (Warehouse B)**: Intermediate buffer storing S1's output
3. **S2 (Set Sorting)**: Pulls from B, routes probabilistically to C1 (40%), C2 (40%), C3 (20%)
4. **S3 (Axis Assembly)**: Pulls from C1, outputs to D1
5. **S4 (Chassis Assembly)**: Pulls from C2, outputs to D2
6. **S5 (Final Assembly)**: Multi-input stage requiring simultaneous pulls from D1, D2, and C3, outputs to E
7. **E (Finished Gliders)**: Final product buffer

### Production Strategy
- **Push Mode**: Orders are released once at t=0 to source stages (S1), system advances based on material availability
- **No CONWIP**: No global WIP cap control
- **Multi-input Synchronization**: S5 requires all inputs (D1, D2, C3) to be available before starting

---

## Stage Parameters

### S1: Type Sorting

| Parameter | Value | Description |
|-----------|-------|-------------|
| `stage_id` | `"S1"` | Stage identifier |
| `name` | `"Type Sorting"` | Stage name |
| `team_id` | `"T1"` | Assigned team |
| `input_buffers` | `[]` | **Source stage, no inputs** |
| `output_buffer` | `"B"` | Output to buffer B |
| `base_process_time_sec` | `2.5` | **Processing time per unit per worker (seconds)** |
| `workers` | `2` | **Number of workers** |
| `time_distribution` | `{"type": "triangular", "p1": 2.0, "p2": 2.5, "p3": 4.0}` | Time distribution: triangular (min=2.0, mode=2.5, max=4.0) |
| `transport_time_sec` | `0.2` | Transport time (seconds) |
| `defect_rate` | `0.00` | Defect rate (0%) |
| `rework_stage_id` | `None` | No rework stage |

**Actual Processing Time Calculation**:
- Base time = `base_process_time_sec / workers` = 2.5 / 2 = 1.25 sec/unit
- Total time = Processing time (sampled from distribution) + Transport time (0.2 sec)
- Example: Sampled value 2.5 sec → Actual = 2.5 / 2 + 0.2 = 1.45 sec/unit

---

### S2: Set Sorting

| Parameter | Value | Description |
|-----------|-------|-------------|
| `stage_id` | `"S2"` | Stage identifier |
| `name` | `"Set Sorting"` | Stage name |
| `team_id` | `"T2"` | Assigned team |
| `input_buffers` | `["B"]` | Pulls from buffer B |
| `output_rules` | `[{"buffer_id": "C1", "p": 0.40}, {"buffer_id": "C2", "p": 0.40}, {"buffer_id": "C3", "p": 0.20}]` | **Probabilistic routing rules** |
| `base_process_time_sec` | `3.0` | Processing time per unit per worker (seconds) |
| `workers` | `2` | Number of workers |
| `time_distribution` | `{"type": "triangular", "p1": 2.0, "p2": 3.0, "p3": 5.0}` | Triangular distribution |
| `transport_time_sec` | `0.3` | Transport time (seconds) |
| `defect_rate` | `0.01` | Defect rate (1%) |
| `rework_stage_id` | `"S2"` | Rework back to self |

**Routing Modes**:
- **Random Mode** (`routing_mode="random"`): Randomly selects output buffer based on probabilities
- **Deterministic Mode** (`routing_mode="deterministic"`): Uses round-robin algorithm to balance outputs, avoiding downstream starvation

**Actual Processing Time**:
- Base time = 3.0 / 2 = 1.5 sec/unit
- Total time = Sampled value / 2 + 0.3 sec

---

### S3: Axis Assembly

| Parameter | Value | Description |
|-----------|-------|-------------|
| `stage_id` | `"S3"` | Stage identifier |
| `name` | `"Axis Assembly"` | Stage name |
| `team_id` | `"T3"` | Assigned team |
| `input_buffers` | `["C1"]` | Pulls from buffer C1 |
| `output_buffer` | `"D1"` | Output to buffer D1 |
| `base_process_time_sec` | `4.0` | Processing time per unit per worker (seconds) |
| `workers` | `2` | Number of workers |
| `time_distribution` | `{"type": "triangular", "p1": 3.0, "p2": 4.0, "p3": 6.0}` | Triangular distribution |
| `transport_time_sec` | `0.4` | Transport time (seconds) |
| `defect_rate` | `0.02` | Defect rate (2%) |
| `rework_stage_id` | `S3` | Rework back to self |

**Actual Processing Time**:
- Base time = 4.0 / 2 = 2.0 sec/unit
- Total time = Sampled value / 2 + 0.4 sec

---

### S4: Chassis Assembly

| Parameter | Value | Description |
|-----------|-------|-------------|
| `stage_id` | `"S4"` | Stage identifier |
| `name` | `"Chassis Assembly"` | Stage name |
| `team_id` | `"T4"` | Assigned team |
| `input_buffers` | `["C2"]` | Pulls from buffer C2 |
| `output_buffer` | `"D2"` | Output to buffer D2 |
| `base_process_time_sec` | `4.0` | Processing time per unit per worker (seconds) |
| `workers` | `2` | Number of workers |
| `time_distribution` | `{"type": "triangular", "p1": 3.0, "p2": 4.0, "p3": 6.0}` | Triangular distribution |
| `transport_time_sec` | `0.4` | Transport time (seconds) |
| `defect_rate` | `0.02` | Defect rate (2%) |
| `rework_stage_id` | `S4` | Rework back to self |

**Actual Processing Time**:
- Base time = 4.0 / 2 = 2.0 sec/unit
- Total time = Sampled value / 2 + 0.4 sec

---

### S5: Final Assembly

| Parameter | Value | Description |
|-----------|-------|-------------|
| `stage_id` | `"S5"` | Stage identifier |
| `name` | `"Final Assembly"` | Stage name |
| `team_id` | `"T5"` | Assigned team |
| `input_buffers` | `["D1", "D2", "C3"]` | **Multi-input: requires simultaneous pulls from D1, D2, and C3** |
| `output_buffer` | `"E"` | Output to final buffer E |
| `base_process_time_sec` | `6.0` | Processing time per unit per worker (seconds) |
| `workers` | `3` | Number of workers |
| `time_distribution` | `{"type": "triangular", "p1": 5.0, "p2": 6.0, "p3": 9.0}` | Triangular distribution |
| `transport_time_sec` | `0.5` | Transport time (seconds) |
| `defect_rate` | `0.03` | Defect rate (3%) |
| `rework_stage_id` | `S5` | Rework back to self |

**Multi-input Synchronization Logic**:
- S5 can only start processing when **all input buffers have inventory**
- If any input is insufficient, the stage enters **starvation state**, waiting for all inputs to be ready

**Actual Processing Time**:
- Base time = 6.0 / 3 = 2.0 sec/unit
- Total time = Sampled value / 3 + 0.5 sec

---

## Buffer Parameters

| Buffer ID | Name | Capacity | Initial Stock | Description |
|----------|------|----------|---------------|-------------|
| `B` | Warehouse B (post-TypeSorting) | 999 | 30 | Output buffer for S1 |
| `C1` | C1 (Axis Parts) | 999 | 0 | S2 output (40%), S3 input |
| `C2` | C2 (Chassis Parts) | 999 | 0 | S2 output (40%), S4 input |
| `C3` | C3 (Final Assembly Only Parts) | 999 | 0 | S2 output (20%), S5 input |
| `D1` | D1 (Axis Subassembly) | 999 | 0 | S3 output, S5 input |
| `D2` | D2 (Chassis Subassembly) | 999 | 0 | S4 output, S5 input |
| `E` | E (Finished Gliders) | 999 | 0 | **Final product buffer** (finished_buffer_ids) |

**Buffer Operations**:
- `can_pull(qty)`: Check if sufficient inventory is available to pull
- `pull(qty)`: Pull specified quantity from buffer
- `can_push(qty)`: Check if sufficient capacity is available to push
- `push(qty)`: Push specified quantity to buffer
- Capacity of `None` indicates infinite capacity

---

## Team Parameters

| Team ID | Name | Size | Shift | Description |
|---------|------|------|-------|-------------|
| `T1` | Type Sorting Team | 2 | day | Assigned to S1 |
| `T2` | Set Sorting Team | 2 | day | Assigned to S2 |
| `T3` | Axis Team | 2 | day | Assigned to S3 |
| `T4` | Chassis Team | 2 | day | Assigned to S4 |
| `T5` | Final Assembly Team | 3 | day | Assigned to S5 |

**Team Utilization Tracking**:
- `busy_time`: Accumulated busy time (simulation time units)
- `last_busy_start`: Current busy start time
- Utilization = `busy_time / sim_time`

**Note**: Team size (`size`) is for visibility only; actual processing speed is controlled by the stage's `workers` parameter.

---

## Key Variables

### Simulation State Variables

| Variable Name | Type | Description |
|---------------|------|-------------|
| `self.t` | `float` | **Current simulation time (seconds)** |
| `self._queue` | `List[Event]` | **Event queue** (priority queue, sorted by time) |
| `self._evt_seq` | `int` | Event sequence number (for breaking time ties) |

### Buffer State

| Variable Name | Type | Description |
|---------------|------|-------------|
| `buffer.current` | `int` | **Current inventory level** |
| `buffer.capacity` | `Optional[int]` | Capacity limit (None = infinite) |

### Stage State

| Variable Name | Type | Description |
|---------------|------|-------------|
| `stage.busy` | `bool` | **Stage busy status** (True = processing, False = idle) |
| `stage.workers` | `int` | **Number of workers** (affects processing speed) |

### WIP and Order Tracking

| Variable Name | Type | Description |
|---------------|------|-------------|
| `self.current_wip` | `int` | **Current work-in-progress** (released but unfinished orders) |
| `self.started` | `int` | Total released orders |
| `self.finished` | `int` | **Completed products** (pushed to finished_buffer_ids) |
| `self.source_stage_orders` | `Dict[str, int]` | **Pending orders for source stages** (key=stage_id, value=order_count) |

### KPI Accumulation Variables

| Variable Name | Type | Description |
|---------------|------|-------------|
| `self.lead_times` | `List[float]` | Lead time for each completed product (seconds) |
| `self.wip_time_area` | `float` | WIP×time integral (for calculating average WIP) |
| `self.last_wip_time` | `float` | Last WIP update time |

### Stage-level Counters

| Variable Name | Type | Description |
|---------------|------|-------------|
| `self.stage_completed_counts` | `Dict[str, int]` | **Completion count for each stage** |
| `self.starvation_counts` | `Dict[str, int]` | **Starvation count for each stage** (waiting for inputs) |
| `self.blocking_counts` | `Dict[str, int]` | **Blocking count for each stage** (output buffer full) |

### Routing State (Deterministic Mode)

| Variable Name | Type | Description |
|---------------|------|-------------|
| `self._served_counts` | `Dict[str, Dict[str, int]]` | Service count per stage per output buffer |
| `self._rr_index` | `Dict[str, int]` | Round-robin index per stage |

### Timeline Sampling

| Variable Name | Type | Description |
|---------------|------|-------------|
| `self.timeline` | `List[Dict[str, Any]]` | **Time series snapshots** (WIP, finished count, throughput, buffer levels at each sample point) |
| `self._sample_dt` | `float` | Sampling interval (seconds, default 5.0) |
| `self._next_sample_t` | `float` | Next sampling time |

---

## Event Types and Logic

### Event Types

| Event Type | Description | Trigger Condition |
|------------|-------------|-------------------|
| `try_start` | Attempt to start processing at a stage | Order release, stage completion, downstream wake-up |
| `complete` | Stage completes processing | Processing time expires |
| `time_advance` | Time advance (internal) | Automatically triggered during each event processing |

### Event Processing Flow

#### 1. `try_start` Event Handler (`_on_try_start`)

**Logic Steps**:
1. Check if stage is busy → If busy, retry after 0.001 sec delay
2. **Source Stage Check**: If no input buffers, check `source_stage_orders[stage_id] > 0`, otherwise return
3. **Multi-input Check**: For each `input_buffers`, attempt to pull 1 unit from buffer
   - If any input is insufficient → Rollback already pulled inputs, increment `starvation_counts`, retry after 0.5 sec delay
4. **Start Processing**:
   - Mark `stage.busy = True`
   - Start team utilization tracking (`team.start_busy(t)`)
   - Calculate processing time: `base_process_time_sec / max(1, workers)` → Sample from distribution → Add transport time
   - Check random disruption (missing bricks) → If occurs, add penalty time
   - Schedule `complete` event at `t + processing_time`

#### 2. `complete` Event Handler (`_on_complete`)

**Logic Steps**:
1. Release team (`team.stop_busy(t)`)
2. **Defect Handling**:
   - If defect occurs (probability = `defect_rate`):
   - Add self.stage_defect_counts,
     - If `rework_stage_id` exists → Trigger `try_start` at rework stage
     - Otherwise → Scrap (`current_wip -= 1`)
   - If no defect → Continue to output
3. **Output Routing**:
   - If `output_buffer` exists → Use that buffer
   - If `output_rules` exists:
     - **Random Mode**: Randomly select based on probabilities
     - **Deterministic Mode**: Use balancing algorithm (select output with minimum served_count/probability ratio)
4. **Push Output**:
   - Attempt to push to selected output buffer
   - If buffer is full → Increment `blocking_counts`, retry `complete` after 0.5 sec delay
   - If successful:
     - If output buffer is in `finished_buffers` → `finished += 1`, `current_wip -= 1`, record lead time
     - Increment `stage_completed_counts[stage_id]`
     - Update routing service count (deterministic mode)
5. **Wake Downstream**:
   - Mark `stage.busy = False`
   - **Source Stage**: If `source_stage_orders[stage_id] > 0`, immediately trigger `try_start`
   - **Non-source Stage**: Immediately trigger `try_start` (attempt to process next item)
   - **Downstream Stages**: For stages consuming the selected output buffer, if they are **not busy**, trigger their `try_start`

#### 3. Shift Constraints

- If event time is outside shift window → Postpone to next shift start time
- Shift checks: `_is_in_shift(t)` and `_advance_to_next_shift_start(t)`

---

## KPI Metrics

### Core KPIs

| KPI | Calculation Formula | Description |
|-----|---------------------|-------------|
| `throughput_per_sec` | `finished / sim_time` | **Throughput** (units/sec) |
| `lead_time_avg_sec` | `sum(lead_times) / len(lead_times)` | **Average lead time** (seconds) |
| `wip_avg_units` | `wip_time_area / sim_time` | **Average work-in-progress** (units) |
| `utilization_per_team` | `team.busy_time / sim_time` | **Team utilization** (per team) |
| `finished_units` | `self.finished` | **Completed products** |
| `started_units` | `self.started` | **Released orders** |
| `service_level` | `self.finished / self.started` | **Service level achieved** |
| `defect_rate_per_stage` | `self.stage_defect_counts / self.stage_completed_counts + self.stage_defect_counts` | **Defect rate for each stage** |
| `total_defect_rate` | `total_defects / total_processed` | **Total production defect rate**|

### Stage-level KPIs

| KPI | Description |
|-----|-------------|
| `stage_completed_counts` | **Completion count for each stage** (dictionary: `{stage_id: count}`) |
| `starvation_counts` | **Starvation count for each stage** (waiting for input materials) |
| `blocking_counts` | **Blocking count for each stage** (output buffer full, cannot push) |
| `defect_defect_counts` | **Defect count for each stage** |

### Time Series Data (`timeline`)

Each sample point contains:
- `t`: Timestamp (seconds)
- `wip`: Current work-in-progress count
- `finished`: Cumulative completed count
- `throughput_per_min`: Throughput per minute (since last sample)
- `B`, `C1`, `C2`, `C3`, `D1`, `D2`, `E`: Current inventory levels for each buffer

---

## Global Parameters

### Simulation Parameters (`parameters`)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `target_takt_sec` | `10.0` | Target takt time (seconds) |
| `timeline_sample_dt_sec` | `5.0` | Timeline sampling interval (seconds) |
| `finished_buffer_ids` | `["E"]` | **Finished product buffer list** (products pushed to these buffers count toward `finished`) |
| `routing_mode` | `"random"` | **Routing mode** (`"random"` or `"deterministic"`) |

### Shift Schedule (`shift_schedule`)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `shift_id` | `"day"` | Shift identifier |
| `start_minute` | `480` | Start time (minutes in day, 8:00 = 480) |
| `end_minute` | `960` | End time (minutes in day, 16:00 = 960) |

### Random Disruptions (`random_events`)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `missing_brick_prob` | `0.10` | **Missing brick probability** (per operation) |
| `missing_brick_penalty_sec` | `2.0` | **Missing brick penalty time** (seconds, additional processing time) |

---

## Time Distribution Types

Supported time distribution types (`time_distribution.type`):

| Type | Parameters | Description |
|------|------------|-------------|
| `constant` | - | Constant time (`base_process_time_sec / workers`) |
| `triangular` | `p1=low, p2=mode, p3=high` | Triangular distribution |
| `normal` | `p1=μ, p2=σ` | Normal distribution |
| `lognormal` | `p1=μ, p2=σ` | Lognormal distribution |
| `uniform` | `p1=a, p2=b` | Uniform distribution |
| `exponential` | - | Exponential distribution (λ = 1 / base_time) |

**Note**: The base value for all distributions is `base_process_time_sec / workers` (time per worker).

---

## Processing Time Calculation Formula Summary

For stage *i*, the actual processing time *S<sub>i</sub>* is calculated as:

```
S_i = T_i + Δ_i + Z_i
```

where:
- **T<sub>i</sub>**: Processing time sampled from distribution `D_i`, with base value `τ_i / w_i`
  - `τ_i` = `base_process_time_sec` (time per unit per worker)
  - `w_i` = `workers` (number of workers)
- **Δ<sub>i</sub>**: `transport_time_sec` (transport time, unaffected by number of workers)
- **Z<sub>i</sub>**: Random disruption penalty (probability = `missing_brick_prob`, penalty = `missing_brick_penalty_sec`)

**Example**:
- S1: `base_process_time_sec=2.5`, `workers=2`, `transport_time_sec=0.2`
  - Base processing time = 2.5 / 2 = 1.25 sec/unit
  - Sample from triangular distribution (e.g., 2.5 sec) → Actual = 2.5 / 2 = 1.25 sec
  - Total time = 1.25 + 0.2 = 1.45 sec/unit

---

## Key Design Decisions

1. **Push Mode**: Orders released once at t=0, system advances based on material availability
2. **Workers Parameter**: Processing time inversely proportional to number of workers (`time_per_unit = base_time / workers`)
3. **Multi-input Synchronization**: S5 requires all inputs (D1, D2, C3) to be available
4. **Deterministic Routing**: Optional mode using balancing algorithm to avoid downstream starvation
5. **Source Stage Order Tracking**: Uses `source_stage_orders` to track pending orders, avoiding infinite retries
6. **Finished Product Identification**: Products pushed to `finished_buffer_ids` count toward `finished`
7. **Shift Constraints**: Events automatically postponed to next shift start time

---

## Usage Recommendations

1. **Adjust Worker Count**: Balance capacity and cost through `workers` parameter
2. **Monitor Starvation/Blocking**: Use `starvation_counts` and `blocking_counts` to identify bottlenecks
3. **Route Balancing**: Use deterministic routing mode at S2 to avoid C1/C2/C3 imbalance
4. **Time Distribution**: Use triangular distribution to model actual variability
5. **Initial Inventory**: Set initial inventory for B (e.g., 30) to quickly start downstream processes

---

*Document Version: v1.0*  
*Last Updated: 2024*

