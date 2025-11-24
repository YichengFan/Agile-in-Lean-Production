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

### • **Service level** (future KPI — will be implemented next stage)  
Definition (conceptual):  
on-time fraction of completed jobs  

Meaning:  
Service level reflects reliability of delivery against due dates.  
Although not implemented yet, it is essential for evaluating customer-oriented performance.

---

### • **Blocking / starvation counts**  
Meaning:  
These indicate structural flow imbalance:

- **Starvation**: a stage wants to start but upstream does not provide input  
- **Blocking**: a stage finishes but downstream buffer is full  

These KPIs help diagnose bottleneck interactions, buffer sizing issues, and routing problems.

---

### • **Defect rate** (future KPI — will be implemented next stage)  
Formula (conceptual):  
defect_rate = defective / total_processed  

Meaning:  
Although the simulation includes defect + rework behavior, it does **not yet** compute a formal defect-rate KPI.  
In future versions, this KPI will quantify quality performance and its impact on throughput and rework load.

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
- Routing split at S₂ (P<sub>2→C1</sub>, P<sub>2→C2</sub>, P<sub>2→C3</sub>)  
- Release policy (push vs pull)  
- Time variance reduction (c<sub>s</sub><sup>2</sup>)  
- Layout / transport reduction (δ<sub>i</sub>)

---

## 9) Calibration → `env.py`

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

## 10) Sanity Targets

- ρ<sub>i</sub> &lt; 0.85 for all stages  
- Balanced routing P<sub>2→C1</sub>=P<sub>2→C2</sub>  
- Start with small K and increase until throughput target is met  

---

## Next Steps

1. Choose policy: Pull (CONWIP K).  
2. Fill numbers: τ<sub>i</sub>, δ<sub>i</sub>, q<sub>i</sub>, π<sub>miss</sub>, m, routing, shifts.  
3. Run `env.py` → KPIs, find bottleneck.  
4. Apply Lean improvements and re-run for comparison.
