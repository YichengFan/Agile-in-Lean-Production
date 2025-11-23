#first-report
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
- Randomness is enabled by default; you can toggle Deterministic processing in the UI.
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

- **Nominal processing time** at stage *i*: τ<sub>i</sub> &gt; 0 (seconds **per worker**)  
- **Number of workers** at stage *i*: w<sub>i</sub> ∈ ℕ<sup>+</sup>  
- **Effective processing time**: τ<sub>i</sub> / w<sub>i</sub>  
- **Transport time**: δ<sub>i</sub> ≥ 0  
- **Time distribution descriptor**: D<sub>i</sub> = (type, p<sub>i1</sub>, p<sub>i2</sub>, p<sub>i3</sub>)  
- **Defect probability**: q<sub>i</sub> ∈ [0, 1]  
- **Rework routing**: defective items at *i* go to r(i) ∈ S or are scrapped  
- **Routing probabilities (S₂)**: P<sub>2→C1</sub>, P<sub>2→C2</sub>, P<sub>2→C3</sub> with Σp = 1  
- **Buffer capacity**: cap<sub>b</sub> ∈ ℕ ∪ {∞}  
- **Shift availability** α(t) ∈ {0, 1} (1 = working, 0 = off)  
- **Random disruption (missing bricks)**:  
  - Occurrence π<sub>miss</sub>  
  - Extra time M with E[M] = m  
- **System WIP cap (CONWIP)** K ∈ ℕ  
- **Release rate** λ orders/sec (for push)

## Global Parameters (`parameters`)

These are configuration-level knobs in `env.py`’s `CONFIG["parameters"]` that affect the whole simulation rather than a single stage.

### Simulation Parameters (`parameters`)

| Parameter                | Default Value | Description |
|--------------------------|---------------|-------------|
| `target_takt_sec`        | `10.0`        | Target takt time (seconds per finished unit). Used as a reference only; the engine does not enforce it. |
| `timeline_sample_dt_sec` | `5.0`         | Sampling interval (seconds) for recording the time series `env.timeline` used in Streamlit charts. |
| `finished_buffer_ids`    | `["E"]`       | List of buffer IDs that count as **finished product**. When a job is pushed into any of these buffers, `finished_units` increments. |
| `routing_mode`           | `"random"`    | Routing mode for stages with `output_rules` (currently S₂). `"random"` uses probabilistic routing each job; `"deterministic"` balances long-run counts to match the target split. |


### Shift Schedule (`shift_schedule`)

| Parameter      | Default Value | Description |
|----------------|---------------|-------------|
| `shift_id`     | `"day"`       | Identifier for the shift (referenced by teams). |
| `start_minute` | `480`         | Start time in minutes from midnight (480 = 08:00). |
| `end_minute`   | `960`         | End time in minutes from midnight (960 = 16:00). |

Shifts repeat every 24 hours. If the current simulation time is outside any shift, processing is paused and events are postponed to the next shift start.

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
- If CONWIP: allow if L(t) &lt; K  
- Else: release per λ or manual  
- Each release triggers “try_start” at a source stage; L(t) += 1

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

- **Throughput** θ = lim C(t)/t  
- **Average WIP** L̄ = (1/t)∫₀ᵗ L(s) ds  
- **Average lead time** W̄ with L̄ = θ·W̄ (Little’s Law)  
- **Utilization** ρ<sub>i</sub> = U<sub>i</sub>(t)/t  
- **Service level**: on-time fraction  
- **Blocking/starvation counts**  
- **Defect rate** = defective / total processed  



---

## 7) Calibration → `env.py`

| Model Element | env.py Mapping |
|----------------|----------------|
| τ<sub>i</sub>, D<sub>i</sub> | `base_process_time_sec`, `time_distribution` |
| w<sub>i</sub> | `workers` |
| δ<sub>i</sub> | `transport_time_sec` |
| q<sub>i</sub>, r(i) | `defect_rate`, `rework_stage_id` |
| P<sub>routing</sub> | `output_rules` (e.g. S2→C1/C2/C3) |
| Multi-input | `input_buffers=["D1","D2","C3"]` |
| α(t) | `shift_schedule` |
| π<sub>miss</sub>, m | `random_events` |
| K | `parameters.conwip_cap` |

---
## 8) Model Assumptions (as implemented)

At the level of this project we rely on a few key simplifications:

1. **Push strategy (no CONWIP loop)**  
   - Orders are released using a **pure push** policy: the UI calls `enqueue_orders(qty)` to inject a batch of jobs into the system.  
   - The theoretical CONWIP WIP cap \(K\) and λ-based release are not active in the current implementation.

2. **No switching cost / setup time**  
   - We do not model product changeovers or sequence-dependent setups.  
   - All jobs are treated as the same “glider” product; routing split at S₂ (to C₁/C₂/C₃) is assumed to be free apart from the processing times already specified.

3. **Simplified timing rule**  
   - For each stage, all detailed activities (walking, picking, assembly, inspection, etc.) are collapsed into a single **service time** random variable:  
     - base processing time drawn from the chosen distribution,  
     - plus a fixed transport time,  
     - plus an optional fixed disruption penalty (“missing bricks”).  
   - Within a stage there is no preemption or overlapping tasks: one job is processed at a time.

4. **Batch size = 1**  
   - Every operation moves exactly **one unit** through the system (one set / glider per start–complete cycle).  
   - Buffers store individual units; there is no lot batching or partial transfer.

5. **Stochastic vs deterministic environment**  
   - By default the simulator is **stochastic**: processing times follow their specified distributions, S₂ can route jobs randomly, and disruptions occur with probability `missing_brick_prob`.  
   - For many experiments we also run a **deterministic configuration** by:
     - switching all time distributions to `constant`,  
     - setting disruption probability to zero, and  
     - optionally enabling *Deterministic routing* at S₂ in the UI.  
   - In this mode the DES behaves as a fully deterministic flow line.

Additional implicit assumptions:

- Each stage behaves like a **single server**: at most one job in process at a time; extra workers only speed up that job (τᵢ / wᵢ).  
- There is effectively infinite raw material supply before S₁; shortages only occur in modelled buffers.  
- Buffer capacities are finite but large by default; blocking happens only when a buffer hits its explicit capacity.

---

## Next Steps

-1.Improving push system
    -**Batch inflow/outflow
    -**Conceptual model improvement
-2.Preparation for pull system - Conceptual modeling
-3.Laying foundation for multiple model production

