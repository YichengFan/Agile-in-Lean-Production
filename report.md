## Project Overview

In this project we build a discrete-event simulator for a LEGO-based lean production line.  
The model represents five stages (Type Sorting, Set Sorting, Axis Assembly, Chassis Assembly, Final Assembly) and seven buffers, with one team assigned to each stage.  
The current implementation focuses on a push system without CONWIP: orders are released in batches and flow through the system according to processing times, routing rules and random disruptions.  
Our goal in this first iteration is to (i) implement a transparent DES engine, (ii) encode the LEGO line as a configurable environment in `env.py`, and (iii) define KPIs such as throughput, WIP, lead time and utilization that we will use in later experiments.  

# LEGO Lean Production — Simulator (Push, DES)

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

Notes
- We currently use push strategy (no CONWIP). Use the sidebar to set orders to release.
- Randomness is enabled by default; we can toggle Deterministic processing in the UI.
- `Environment.xlsx` is no longer used at runtime; parameters live in `env.py`'s `CONFIG`.

---

# Mathematical Model (v0.1)

## 1) Sets, Indices, Graph

- **Stages (processing nodes)**: S<sub>1–5</sub> = {S<sub>1</sub>, S<sub>2</sub>, S<sub>3</sub>, S<sub>4</sub>, S<sub>5</sub>}
  - S<sub>1</sub>: Type Sorting  
  - S<sub>2</sub>: Set Sorting  
  - S<sub>3</sub>: Axis Assembly  
  - S<sub>4</sub>: Chassis Assembly  
  - S<sub>5</sub>: Final Assembly
- **Buffers (inventories)**: B = {B, C<sub>1</sub>, C<sub>2</sub>, C<sub>3</sub>, D<sub>1</sub>, D<sub>2</sub>, E}
- **Teams (resources)**: T = {T<sub>1</sub>, T<sub>2</sub>, T<sub>3</sub>, T<sub>4</sub>, T<sub>5</sub>}
- **Arcs**: directed edges between buffers and stages describing the flow:  
  S<sub>1</sub> → B → S<sub>2</sub> → {C<sub>1</sub>, C<sub>2</sub>, C<sub>3</sub>} → (S<sub>3</sub>, S<sub>4</sub>, S<sub>5</sub>) → E

---

## 2) Parameters
These are configuration-level knobs in `env.py`’s `CONFIG["parameters"]` that affect the whole simulation rather than a single stage.

### Operational(simulation) Parameters (`parameters`)

| Parameter                | Default Value | Description |
|--------------------------|---------------|-------------|
| `timeline_sample_dt_min` | `5.0`         | Sampling interval (minutes) for recording the time series `env.timeline` used in Streamlit charts. |
| `finished_buffer_ids`    | `["E"]`       | List of buffer IDs that count as **finished product**. When a job is pushed into any of these buffers, `finished_units` increments. |
| `routing_mode`           | `"random"`    | Routing mode for stages with `output_rules` (currently S₂). `"random"` uses probabilistic routing each job; `"deterministic"` balances long-run counts to match the target split. |


### Random Disruptions (`random_events`)

| Parameter                   | Default Value | Description |
|-----------------------------|---------------|-------------|
| `missing_brick_prob`        | `0.10`        | Per-operation probability that a “missing bricks” disruption occurs. |
| `missing_brick_penalty_sec` | `2.0`         | Extra processing time (seconds) added to that operation when a disruption happens. |

---

## 3) Random Variables and Service Times

For each job at stage *i*:

S<sub>i</sub> = T<sub>i</sub> + Δ<sub>i</sub> + Z<sub>i</sub>

where  
- T<sub>i</sub> ~ D<sub>i</sub>(τ<sub>i</sub> / w<sub>i</sub>; p<sub>i1</sub>, p<sub>i2</sub>, p<sub>i3</sub>) — **processing time scales inversely with workers**  
- Δ<sub>i</sub> = δ<sub>i</sub> — **transport time (unaffected by workers)**  
- Z<sub>i</sub> = M · 1{disruption} with P(disruption)=π<sub>miss</sub>  

Defect outcome: with prob q<sub>i</sub>, route to r(i) or scrap.

**Example**: With τ<sub>i</sub> = 3.0 sec/worker, δ<sub>i</sub> = 0.3 sec:
- 1 worker: E[S<sub>i</sub>] = 3.0 / 1 + 0.3 = 3.3 sec
- 2 workers: E[S<sub>i</sub>] = 3.0 / 2 + 0.3 = 1.8 sec
- 3 workers: E[S<sub>i</sub>] = 3.0 / 3 + 0.3 = 1.3 sec

---

## 4) State Variables

- Buffer X<sub>b</sub>(t) ∈ [0, cap<sub>b</sub>]  
- Stage busy Y<sub>i</sub>(t) ∈ {0, 1}  
- Team busy U<sub>i</sub>(t)  
- WIP L(t) = number of released but unfinished jobs  
- Completions C(t) = finished goods in E  

For S<sub>5</sub> (multi-input): start only if  
X<sub>D1</sub> ≥ 1 and X<sub>D2</sub> ≥ 1 and X<sub>C3</sub> ≥ 1.

---

## 5) Event Logic

**Release event**  
- The current implementation uses a **pure push system**: jobs are released only via manual batch release (`enqueue_orders`).  
- CONWIP (WIP-cap K) and λ-based continuous release are part of the conceptual model but **not implemented** in the current code.

**Try-start at stage i**  
- If Y<sub>i</sub> = 0 and all inputs available:  
  pull 1 unit from each b; set Y<sub>i</sub> = 1; sample S<sub>i</sub>; schedule completion t+S<sub>i</sub>  
- Else: retry after ε

**Completion at stage i**  
- Stop U<sub>i</sub>  
- With prob q<sub>i</sub>: defect → rework or scrap (L–1)  
- Else: push to output; if blocked retry; if sink → finished C+1, L–1  
- Set Y<sub>i</sub> = 0; trigger downstream “try_start”  

Shift constraint: postpone if α(t)=0.

---
## 6) KPIs

Below are the Key Performance Indicators reported by the simulator.  
Each KPI is defined mathematically and explained in terms of why it matters for system performance.

---

### ▪ Throughput (`throughput_per_sec`)

**Formula**

$$
\text{Throughput} = 
\frac{\text{finished units}}{\text{simulation time (sec)}}
$$


**Meaning**

Number of finished units per second, computed as `finished_units / sim_time_sec`.  
In practice we often convert this to units/hour by multiplying by 3600.  
This is the primary indicator of system capacity.

---

### ▪ Average WIP (`wip_avg_units`)

**Formula**

$$
\bar{L} = \frac{1}{t} \int_0^t L(s)\,\mathrm{d}s
$$

where \(L(s)\) is Work-In-Process at time \(s\).

**Meaning**

Time-averaged WIP inside the system.  
Directly linked to lead time via Little’s Law.  
High WIP \(\Rightarrow\) long lead times and unstable flow.

---

### ▪ Average lead time (`lead_time_avg_sec`)

**Formula**

$$
\bar{W} = \frac{1}{N} \sum_{j=1}^{N} 
\bigl(t_j^{\text{finish}} - t_j^{\text{release}}\bigr)
$$

With Little’s Law:

$$
\bar{L} = \theta \cdot \bar{W}
$$

where \(\theta\) is the throughput.

**Meaning**

Average time a job spends in the system from release to completion.  
Shorter and more stable lead times indicate better responsiveness.

---

### ▪ Utilization (`utilization_per_team`)

**Formula**

$$
\rho_i = \frac{U_i(t)}{t}
$$

where \(U_i(t)\) is the total busy time of team \(i\) over horizon \(t\).

**Meaning**

Shows how loaded each workstation/team is.  
High utilization (e.g. \(\rho_i > 0.85\)) indicates a potential bottleneck,  
while very low utilization suggests overstaffing or poor line balancing.



---

### **▪ Bottleneck-Blocking and starvation counts**

**Definitions**

- **Bottlenck(Starvation)**: a stage attempts to start but at least one required input buffer is empty  
- **Bottlenck(Blocking)**: a stage finishes processing but the output buffer is full  

Both are counted per stage.

**Meaning**

These counts diagnose flow imbalance, routing issues, and buffer sizing problems.  
High starvation → upstream slow  
High blocking → downstream constrained

---

### ▪ Defect rate (future KPI)

**Formula**

$$
\text{Defect rate} = \frac{\text{defective jobs}}{\text{total processed jobs}}
$$

**Note**

The simulator already includes defect and rework logic,
but we do not yet compute or output a formal defect-rate KPI.  
This KPI will be implemented in the next development stage.


---

## 7) Calibration → `env.py`

| Model Element | env.py Mapping |
|----------------|----------------|
| τ<sub>i</sub> | `base_process_time_min` (effective time = τ/√workers, constant) |
| w<sub>i</sub> | `workers` |
| δ<sub>i</sub> | `transport_time_min` (or `transport_time_to_outputs_min` for S2→C1/C2/C3) |
| q<sub>i</sub>, r(i) | `defect_rate`, `rework_stage_id` (π<sub>miss</sub>/random_events not in current sim) |
| P<sub>routing</sub> | `output_buffers_by_model` (S2→C1/C2/C3 per model) |
| Multi-input | `input_buffers` (e.g. S5: ["C3","D1","D2"]) |
| K | `parameters.conwip_wip_cap` |

---
## 8) Model Assumptions (as implemented)

At the level of this project we rely on a few key simplifications:

1. **Push and Pull modes**  
   - **Push**: Orders are released from demand forecast at simulation start; no CONWIP cap.  
   - **Pull**: User sets Order Release (quantity per model); CONWIP WIP cap \(K\) and closed-loop auto-release are implemented. When “Enable Pull” is ticked, the UI uses `conwip_wip_cap` and `enqueue_orders(..., is_replenishment=True)` for replenishment.

2. **No switching cost / setup time**  
   - We plan to implement this assumption in the next iteration.
   - We do not model product changeovers or sequence-dependent setups.
   - Multi-model (m01–m04) with model-specific BOMs; routing split at S₂ (to C₁/C₂/C₃) is deterministic per model; no setup time between models.

3. **Simplified timing rule**  
   - For each stage, service time is **constant**: effective_time = base_process_time_min / √workers + transport_time_min (no random distribution in current sim).  
   - Within a stage there is no preemption: one job at a time; workers reduce time via √workers.

4. **Batch size = 1**  
   - Every operation moves exactly **one unit** through the system (one set / glider per start–complete cycle).  
   - Buffers store individual units; there is no lot batching or partial transfer.

5. **Deterministic processing**  
   - The current simulator uses **constant** processing time per stage (effective_time = base_time / √workers); no random time distributions.  
   - S₂ routes to C1/C2/C3 per model-specific outputs; defect/rework is the only stochastic element (if defect_rate > 0).

Additional implicit assumptions:

- Each stage behaves like a **single server**: at most one job in process at a time; extra workers reduce time via τᵢ / √wᵢ.  
- There is effectively infinite raw material supply before S₁; shortages only occur in modelled buffers.  
- Buffer capacities are finite but large by default; blocking happens only when a buffer hits its explicit capacity.

---


