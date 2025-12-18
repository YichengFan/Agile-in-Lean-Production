# LEGO Lean Production — Simulation Model （Pull (CONWIP + Kanban) ）

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
- Updated: We now support Pull control (CONWIP + Kanban) via `env.py` config and the Streamlit panel.
- `Environment.xlsx` is no longer used at runtime; parameters live in `env.py`'s `CONFIG`.
- Randomness is enabled by default; you can toggle Deterministic processing in the UI.

---

## Mathematical Model

For the **mathematical model** (objective function, decision variables, constraints, cost formulations), please refer to [`mathematical_model.md`](mathematical_model.md).

This document focuses on the **simulation model** implementation details.

---

# Simulation Model (Discrete-Event Simulation)

## 1) Sets, Indices, Graph

- **Stages (processing nodes)**: S<sub>1–5</sub> = {S<sub>1</sub>, S<sub>2</sub>, S<sub>3</sub>, S<sub>4</sub>, S<sub>5</sub>}
  - S<sub>1</sub>: Type Sorting  
  - S<sub>2</sub>: Set Sorting  
  - S<sub>3</sub>: Axis Assembly  
  - S<sub>4</sub>: Chassis Assembly  
  - S<sub>5</sub>: Final Assembly
- **Buffers (inventories)**: B = {A,B, C<sub>1</sub>, C<sub>2</sub>, C<sub>3</sub>, D<sub>1</sub>, D<sub>2</sub>, E}
- **Teams (resources)**: T = {T<sub>1</sub>, T<sub>2</sub>, T<sub>3</sub>, T<sub>4</sub>, T<sub>5</sub>}
- **Arcs**: directed edges between buffers and stages describing the flow:  
  S<sub>1</sub> → B → S<sub>2</sub> → {C<sub>1</sub>, C<sub>2</sub>, C<sub>3</sub>} → (S<sub>3</sub>, S<sub>4</sub>, S<sub>5</sub>) → E

---

## 2) Parameters

- **Nominal processing time** at stage *i*: τ<sub>i</sub> &gt; 0 (seconds **per worker**)  
- **Number of workers** at stage *i*: w<sub>i</sub> ∈ ℕ<sup>+</sup>  
- **Effective processing time**: (τ<sub>i</sub> * total_parts) / w<sub>i</sub>
- **Transport time**: δ<sub>i</sub> ≥ 0
- **S2 destination-specific delivery time (optional)**: δ_{2→b} ≥ 0  
  S2 can use a separate delivery delay (transport delay) for each destination buffer b ∈ {C1, C2, C3}.  
  When enabled, deliveries to C1/C2/C3 are scheduled after their own δ_{2→C1}, δ_{2→C2}, δ_{2→C3} (instead of a single shared δ_2).

- **Time distribution descriptor**: D<sub>i</sub> = (type, p<sub>i1</sub>, p<sub>i2</sub>, p<sub>i3</sub>)  
- **Defect probability**: q<sub>i</sub> ∈ [0, 1]  
- **Rework routing**: defective items at *i* go to r(i) ∈ S or are scrapped  
- **Buffer capacity**: cap<sub>b</sub> ∈ ℕ ∪ {∞}  
- **Shift availability** α(t) ∈ {0, 1} (1 = working, 0 = off)  
- **Random disruption (missing bricks)**:  
  - Occurrence π<sub>miss</sub>  
  - Extra time M with E[M] = m  
- **System WIP cap (CONWIP)** K ∈ ℕ  
- **Release rate** λ orders/sec  (**optional; push baseline only**)
**Pull-control parameters (added)**(implementation under `CONFIG["parameters"]` in `env.py`)
 - **Release-stage set**: R ⊆ S (token-gated entry stages)  
  Code: `parameters.release_stage_ids` (e.g., `["S1"]`)
- **CONWIP WIP cap**: K ∈ ℕ (max released-but-unfinished jobs)  
  Code: `parameters.conwip_wip_cap`
- **Closed-loop CONWIP switch**: β ∈ {0,1} (β=1 → release 1 job per finished unit)  
  Code: `parameters.auto_release_conwip`
- **Kanban buffer caps**: κ<sub>b</sub> ∈ ℕ for b ∈ B (local WIP control limits; not physical capacity)  
  Code: `parameters.kanban_caps` (dict: b ↦ κ<sub>b</sub>)


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
**Per-output transport times (added)**  
  For stages that deliver to multiple buffers (e.g., S2 → C1/C2/C3), we support a per-destination transport-time map:

  - **Per-output transport time map**: δ<sub>i→b</sub> ≥ 0 for each destination buffer b  
    Code: `transport_time_to_outputs_sec = {"C1": ..., "C2": ..., "C3": ...}`

  - **Event logic**: one `deliver` event per destination buffer (rather than a single combined transport).  
    Code path: `_on_complete()` schedules multiple `deliver` events; `_on_deliver()` performs the actual push into each output buffer.


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

**Release event (Pull: CONWIP + Kanban)**  
- Release is **token-based** to the entry stage(s) R.  
- If CONWIP is enabled: allow release only if L(t) < K.  
- On release: issue tokens to stages in R, append release timestamp(s), and set L(t) += 1.  
- If closed-loop CONWIP is enabled (β = 1): each finished unit triggers releasing one new order (subject to K).

---

### Try-start at stage i (pull-gated, BOM deterministic)

A stage i can start only if all of the following hold:

- **Idle condition:** Y<sub>i</sub>(t) = 0 (`stage.busy == False`)
- **Release-stage token gating (CONWIP):** if i ∈ R (`release_stage_ids`), then  
  `stage_orders[i] > 0` must hold, and starting consumes one token.
- **Kanban gating:** for any controlled output buffer b with cap K<sub>b</sub> (`kanban_caps[b]`), the projected post-push level must satisfy:  
  X<sub>b</sub>(t) + ΔX<sub>b</sub> ≤ K<sub>b</sub>  
  Otherwise, the stage does not start and retries later.
- **Material availability (BOM):** for each required item k with quantity a<sub>ik</sub>, required inputs must exist in the listed input buffers; otherwise the stage starves and retries.

If all checks pass:
- Pull BOM items from buffers
- Set Y<sub>i</sub>(t) = 1
- Sample processing time and schedule `complete` at t + S<sub>i</sub>

---

### Completion at stage i (updated)

On `complete(i)`:
- Stop team utilization timer U<sub>i</sub>
- **Defect handling:** with probability q<sub>i</sub>:
  - route to rework r(i), or
  - scrap (and reduce WIP by 1 if it leaves the system)
- **Otherwise push deterministic outputs** (`output_buffers`)
  - If downstream is blocked, retry later (no partial commit)
  - If output goes to finished buffer E:
    - finished count increases
    - WIP decreases (job leaves the system)
    - If `auto_release_conwip=True`, release one new order (closed-loop)

---

### Deliver event (added: per-output transport)

When a stage i has per-destination transport times δ<sub>i→b</sub>:

- `complete(i)` schedules one `deliver(i→b)` per output buffer b
- Each deliver occurs after its own delay δ<sub>i→b</sub>
- The stage is freed only after **all** deliveries succeed (multi-delivery completion)

---

Shift constraint: postpone any event if α(t)=0.

## 6) KPIs

Below are the Key Performance Indicators (KPIs) we use to evaluate the performance of the LEGO production simulator.  
Each KPI includes a simple mathematical definition and a short explanation of why it matters.

---

### • **Throughput**  
Formula:  
θ = lim C(t) / t  

Meaning:  
Throughput measures how many finished units the system produces per unit time.  
It is the primary indicator of system capacity and directly determined by the bottleneck stage.

---

### • **Average WIP**  
Formula:  
L̄ = (1 / t) ∫₀ᵗ L(s) ds  

Meaning:  
Average work-in-process reflects system congestion.  
High WIP → longer lead times and more variability.  
It is directly tied to lead time through Little’s Law.

---

### • **Average lead time**  
Relationship (Little’s Law):  
L̄ = θ · W̄  

Meaning:  
Average time a job spends inside the system from release to completion.  
Shorter and more stable lead times mean better responsiveness and smoother flow.

---

### • **Utilization**  
Formula:  
ρᵢ = Uᵢ(t) / t  

Meaning:  
Measures how busy each workstation/team is.  
High utilization (especially > 0.85) suggests a potential bottleneck and affects throughput stability.

---

### • **Service level**
Formula: SL = N<sub>fin</sub>(t) / N<sub>target</sub>

Meaning:  
Service level reflects reliability of delivery against planned amount.


---

### • **Blocking / starvation counts**  
Meaning:  
These indicate structural flow imbalance:

- **Starvation**: a stage wants to start but upstream does not provide input  
- **Blocking**: a stage finishes but downstream buffer is full  

These KPIs help diagnose bottleneck interactions, buffer sizing issues, and routing problems.

---
### • **Kanban blocking counts (added)**

Meaning:  
Counts how often a stage wants to start but is stopped because starting would push a controlled buffer above its Kanban cap.

- **Kanban blocking**: a stage wants to start but local WIP limit (`kanban_caps`) would be exceeded

---

### • **Defect rate**
Formula: DR<sub>i</sub> = N<sub>i,def</sub> / (N<sub>i,def</sub> + N<sub>i,ok</sub>); DR<sub>total</sub> = Σ N<sub>i,def</sub> / Σ (N<sub>i,def</sub> + N<sub>i,ok</sub>)


Meaning:  
Defect in each stage will be recorded, defect rate for each stage will be computed as well as the total defect rate, thus easier for tracking performances.

---

## 7) Analytical Approximations

Each stage ≈ G/G/1 queue with μ<sub>i</sub> = 1/E[S<sub>i</sub>].  

Kingman approximation:  

W<sub>q,i</sub> ≈ ρ<sub>i</sub> / (1–ρ<sub>i</sub>) × (c<sub>a,i</sub><sup>2</sup>+c<sub>s,i</sub><sup>2</sup>)/2 × 1/μ<sub>i</sub>  

Then W<sub>i</sub> ≈ W<sub>q,i</sub> + 1/μ<sub>i</sub>.  

Bottleneck: max ρ<sub>i</sub> limits throughput θ.  

CONWIP: K controls θ(K) and W̄(K)=L̄(K)/θ(K). Tune K to balance throughput/lead time.

---

## 8) Optimization Knobs

- Staffing (team sizes)  
- WIP caps (global K or per-buffer Kanban)  
- Release policy (push vs pull)  
- Time variance reduction (c<sub>s</sub><sup>2</sup>)  
- Layout / transport reduction (δ<sub>i</sub>)

---

## 9) Simulation Implementation Mapping

| Simulation Element | `env.py` Implementation |
|----------------|----------------|
| τ<sub>i</sub>, D<sub>i</sub> | `base_process_time_sec`, `time_distribution` |
| w<sub>i</sub> | `workers` |
| δ<sub>i</sub> | `transport_time_sec` |
| q<sub>i</sub>, r(i) | `defect_rate`, `rework_stage_id` |
| BOM inputs | `required_materials` + `input_buffers` |
| BOM outputs | `output_buffers` (deterministic itemized outputs) |
| Multi-input | `input_buffers=["D1","D2","C3"]` |
| α(t) | `shift_schedule` |
| π<sub>miss</sub>, m | `random_events` |
| CONWIP cap K | `parameters.conwip_wip_cap` |
| Release stages R | `parameters.release_stage_ids` |
| Closed-loop CONWIP β | `parameters.auto_release_conwip` |
| Kanban caps κ<sub>b</sub> | `parameters.kanban_caps` |
| WIP L(t) | `self.current_wip` |
| Lead time timestamps | `self._release_times` |
| Kanban blocking | `self.kanban_blocking_counts` |
| Release tokens | `self.stage_orders` |
| δ<sub>i→b</sub> (per-output transport) | `stages[].transport_time_to_outputs_sec` + `deliver` event (`_on_deliver`) |
| N<sub>fin</sub>(t), N<sub>target</sub> | `self.finished`, `self.started` |
| N<sub>i,def</sub>, N<sub>i,ok</sub> | `stage_defect_counts`, `stage_completed_counts` |

---

## 10) Simulation Assumptions

- **Discrete-event simulation**: Events processed in chronological order using a priority queue
- **Pull strategy (CONWIP + Kanban)**: Order release is token-gated at release stage(s) and constrained by a global CONWIP WIP cap
- **Closed-loop CONWIP (optional)**: If enabled, each finished unit releases one new order (subject to the WIP cap)
- **Event-driven**: `try_start`, `complete`, and (optional) `deliver` events drive state transitions
- **Time sampling**: KPIs sampled at intervals (`timeline_sample_dt_sec`) for visualization
- **Random number generation**: Uses Python `random` module with optional seed
- **Deterministic BOM flow**: No probabilistic routing; outputs are itemized and deterministic

---

## 11) Cost & Profit KPIs (implemented)

- Revenue = `unit_price` × finished (or capped by `demand_qty` if provided)
- Material cost = `unit_material_cost` × actual consumed quantity (accumulated per BOM pull)
- Labor cost = Σ (team busy time × team size × `labor_costs_per_team_sec`)
- Inventory holding cost = Σ (∫inventory dt × `holding_costs_per_buffer_sec`)
- Profit = Revenue − (Material + Labor + Inventory + Other)
- Configure under `parameters.cost` in `CONFIG`.

---

## 12) Simulation Validation Targets

- ρ<sub>i</sub> &lt; 0.85 for all stages  
- Throughput matches analytical estimates (Kingman approximation)
- Little's Law holds: L̄ = θ · W̄

---

## 13) Next Steps

1. **Simulation**: Run `env.py` → collect KPIs, identify bottlenecks
2. **Mathematical Model**: Use [`mathematical_model.md`](mathematical_model.md) to formulate optimization problem
3. **Optimization**: Determine optimal Q, w<sub>i</sub>, cap<sub>b</sub>
4. **Lean Transition**: Apply waste elimination → reduce costs → increase profit
5. **Validation**: Compare simulation results with mathematical model predictions
