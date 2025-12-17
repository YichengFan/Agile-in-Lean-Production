# Mathematical Model for LEGO Lean Production System

## 1. Objective Function

### Primary Objective: Maximize Profit

**Maximize profit:** Π = R − C

where:
- **R** = Revenue
- **C** = Total Cost

### Revenue Model

R = p · E[min(Q, D)]

where:
- **p** = Selling price per finished glider (€/unit)
- **Q** = Produced finished goods quantity (units)
- **D** = Random demand (units)
- **E[min(Q, D)]** = Expected sales quantity (accounting for demand uncertainty)

**Assumptions:**
- Demand is stochastic with known distribution
- Unsold finished goods have zero salvage value (or can be modeled separately)
- All produced units that meet demand are sold at price p

---

## 2. Cost Components

### Total Cost Decomposition

 C = C<sub>material</sub> + C<sub>processing</sub> + C<sub>inventory</sub> + C<sub>other</sub>

where:
- **C<sub>material</sub>** = Material costs (bricks)
- **C<sub>processing</sub>** = Processing costs (labor)
- **C<sub>inventory</sub>** = Inventory holding costs
- **C<sub>other</sub>** = Other costs (defects, rework, disruptions, etc.)

---

## 3. Material Costs

### Material Cost Model

 C<sub>material</sub> = c<sub>b</sub> · N<sub>b</sub>(Q)

where:
- **c<sub>b</sub>** = Cost per brick (€/brick)
- **N<sub>b</sub>(Q)** = Total number of bricks required to produce Q finished gliders

**Accounting note (simulation implementation):** material cost is accumulated on actual consumption (each BOM pull), not on order release.

### Brick Requirements (BOM-driven)

N<sub>b</sub>(Q) = ∑<sub>i=1..m</sub> ∑<sub>k∈𝒦</sub> m<sub>i,k</sub> · Q<sub>i</sub>

where:
- **m<sub>i,k</sub>** = Quantity of item/brick *k* consumed per unit at stage *i* (BOM)
- **Q<sub>i</sub>** = Quantity processed at stage *i* (accounting for defects/rework)
- **𝒦** = Set of item types

### Material Cost Constraints

- **Raw material availability**: N<sub>b</sub>(Q) ≤ A<sub>0</sub>, where A<sub>0</sub> is initial raw material stock
- **Material flow conservation**: Material consumed at each stage must equal material received from upstream stages

---

## 4. Processing Costs

### Processing Cost Model

C<sub>processing</sub> = ∑<sub>i=1..m</sub> ( c<sub>l,i</sub> · w<sub>i</sub> · T<sub>i</sub>(Q) )

where:
- **c<sub>l,i</sub>** = Labor cost per worker per unit time at stage i (€/worker·sec)
- **w<sub>i</sub>** = Number of workers at stage i (decision variable)
- **T<sub>i</sub>(Q)** = Total processing time required at stage i to produce Q finished units (sec)

### Processing Time Calculation

For each stage i: T<sub>i</sub>(Q) = ∑<sub>j=1..Q<sub>i</sub></sub> S<sub>i,j</sub>

where:
- **Q<sub>i</sub>** = Number of units processed at stage i
- **S<sub>i,j</sub>** = Service time for unit j at stage i (random variable)

### Service Time Model

S<sub>i,j</sub> = τ<sub>i</sub> / w<sub>i</sub> + Δ<sub>i</sub> + Z<sub>i,j</sub>

where:
- **τ<sub>i</sub>** = Base processing time per worker at stage i (sec/worker)
- **w<sub>i</sub>** = Number of workers (affects processing speed)
- **Δ<sub>i</sub>** = Transport time at stage i (sec, unaffected by workers)
- **Z<sub>i,j</sub>** = Random disruption penalty (e.g., missing bricks)

**Disruption Model:** Z<sub>i,j</sub> = M · 1{disruption occurs}

where:
- **M** = Disruption penalty time (sec)
- **P(disruption)** = π<sub>miss</sub> (missing brick probability)

### Processing Cost Constraints

- **Worker availability**: w<sub>i</sub> ≤ W<sub>max,i</sub> (maximum workers per stage)
- **Shift constraints**: Processing only occurs during shift hours α(t) ∈ {0, 1}
- **Capacity constraints**: Throughput at stage i limited by processing capacity

---

## 5. Inventory Holding Costs

### Inventory Holding Cost Model

C<sub>inventory</sub> = ∑<sub>b∈B</sub> h<sub>b</sub> · X̄<sub>b</sub>(T)

where:
- **h<sub>b</sub>** = Holding cost per unit per unit time at buffer b (€/unit·sec)
- **X̄<sub>b</sub>(T)** = Average inventory level at buffer b over time horizon T
- **B** = Set of all buffers {B, C<sub>1</sub>, C<sub>2</sub>, C<sub>3</sub>, D<sub>1</sub>, D<sub>2</sub>, E}

### Average Inventory Level

X̄<sub>b</sub>(T) = (1/T) ∫₀^T X<sub>b</sub>(t) dt

where:
- **X<sub>b</sub>(t)** = Inventory level at buffer b at time t
- **T** = Time horizon

### Inventory Dynamics

For each buffer b:

dX<sub>b</sub>(t)/dt = λ<sub>b,in</sub>(t) − λ<sub>b,out</sub>(t)

where:
- **λ<sub>b,in</sub>(t)** = Inflow rate to buffer b at time t
- **λ<sub>b,out</sub>(t)** = Outflow rate from buffer b at time t

### Inventory Constraints

- **Buffer capacity**: 0 ≤ X<sub>b</sub>(t) ≤ cap<sub>b</sub> for all t
- **Material flow balance**: Inflow equals outflow at steady state (for each buffer)
- **Initial conditions**: X<sub>b</sub>(0) = X<sub>b,0</sub> (initial stock)

---

## 6. Other Costs

### Defect and Rework Costs

C<sub>defect</sub> = Σ<sub>i=1..m</sub> c<sub>d,i</sub> · Q<sub>i</sub> · q<sub>i</sub> + c<sub>r</sub> · Q<sub>rework</sub>

where:
- **c<sub>d,i</sub>** = Cost per defect at stage i (€/defect)
- **q<sub>i</sub>** = Defect rate at stage i (probability)
- **c<sub>r</sub>** = Rework cost per unit (€/unit)
- **Q<sub>rework</sub>** = Quantity requiring rework

### Disruption Costs

C<sub>disruption</sub> = c<sub>disrupt</sub> · Σ<sub>i=1..m</sub> Q<sub>i</sub> · π<sub>miss,i</sub> · E[M<sub>i</sub>]

where:
- **c<sub>disrupt</sub>** = Cost per unit of disruption time (€/sec)
- **π<sub>miss,i</sub>** = Disruption probability at stage i
- **E[M<sub>i</sub>]** = Expected disruption penalty time at stage i

---

## 7. Decision Variables

### Primary Decision Variables

1. **Q** = Production quantity (finished gliders) - **primary decision**
2. **w<sub>i</sub>** = Number of workers at stage i, i ∈ {1, 2, ..., m}
3. **cap<sub>b</sub>** = Buffer capacity at buffer b, b ∈ B
4. **K** = CONWIP level (if using pull strategy)

### Secondary Decision Variables

- **λ** = Release rate (for push strategy)
- **Shift schedules** α(t)
- **Quality improvement investments** (affecting q<sub>i</sub>)

---

## 8. Constraints

### Capacity Constraints

**Stage capacity:** θ<sub>i</sub> ≤ μ<sub>i</sub>(w<sub>i</sub>) = 1 / (τ<sub>i</sub> / w<sub>i</sub> + Δ<sub>i</sub> + E[Z<sub>i</sub>])

where:
- **θ<sub>i</sub>** = Throughput at stage i
- **μ<sub>i</sub>(w<sub>i</sub>)** = Effective single-server capacity with faster service time from more workers (parallel servers are **not** modeled here)
- **E[Z<sub>i</sub>]** = Expected disruption penalty time (e.g., missing bricks)

**Bottleneck constraint:** θ = min<sub>i</sub> θ<sub>i</sub>

**System throughput:** θ ≤ θ<sub>target</sub>

### Material Flow Constraints

**Conservation of flow:** ∑<sub>b∈inputs(i)</sub> λ<sub>b,out</sub> = ∑<sub>b∈outputs(i)</sub> λ<sub>b,in</sub> = θ<sub>i</sub>

**Multi-input synchronization (for S<sub>5</sub>):** λ<sub>D1,out</sub> = λ<sub>D2,out</sub> = λ<sub>C3,out</sub> = θ<sub>5</sub>

### Inventory Constraints

**Buffer capacity:** 0 ≤ X<sub>b</sub>(t) ≤ cap<sub>b</sub> for all t  
**Non-negativity:** Q ≥ 0, w<sub>i</sub> ≥ 1, cap<sub>b</sub> ≥ 0

### Routing Constraints

Deterministic BOM routing: outputs are fixed item sets per stage; no probabilistic routing is used.

---

## 9. State Variables and Transition Functions

### State Variables

**Buffer states:**
- **X<sub>b</sub>(t)** = Inventory level at buffer b at time t

**Stage states:**
- **Y<sub>i</sub>(t)** ∈ {0, 1} = Busy status of stage i (0 = idle, 1 = busy)
- **U<sub>i</sub>(t)** = Cumulative busy time of stage i up to time t

**System states:**
- **L(t)** = Work-in-process (WIP) at time t
- **C(t)** = Cumulative finished goods count up to time t
- **Q(t)** = Cumulative production quantity up to time t

### State Transition Functions

**Release event (t = t<sub>release</sub>):** L(t<sup>+</sup>) = L(t<sup>−</sup>) + 1; Q<sub>started</sub>(t<sup>+</sup>) = Q<sub>started</sub>(t<sup>−</sup>) + 1  
**Stage start (i at time t):** Y<sub>i</sub>(t<sup>+</sup>) = 1; X<sub>b,in</sub>(t<sup>+</sup>) = X<sub>b,in</sub>(t<sup>−</sup>) − 1 for all b ∈ inputs(i)  
**Stage completion (i at time t):** Y<sub>i</sub>(t<sup>+</sup>) = 0; U<sub>i</sub>(t<sup>+</sup>) = U<sub>i</sub>(t<sup>−</sup>) + S<sub>i,j</sub>  
**Defect with prob q<sub>i</sub>:** if rework → send to r(i); if scrap → L(t<sup>+</sup>) = L(t<sup>−</sup>) − 1  
**Otherwise (good unit):** X<sub>b,out</sub>(t<sup>+</sup>) = X<sub>b,out</sub>(t<sup>−</sup>) + 1 for chosen output buffer b  
**Finished goods (S5 → E):** C(t<sup>+</sup>) = C(t<sup>−</sup>) + 1; L(t<sup>+</sup>) = L(t<sup>−</sup>) − 1; Q(t<sup>+</sup>) = Q(t<sup>−</sup>) + 1

---

## 10. Model Assumptions

### Production System Assumptions

1. **Single product type**: All gliders are identical (or color variants with no structural difference)
2. **Batch size = 1**: Each stage processes one unit at a time
3. **No setup times**: No switching costs between production runs
4. **Deterministic routing (optional)**: Can use deterministic or probabilistic routing
5. **Multi-input synchronization**: S<sub>5</sub> requires all inputs (D<sub>1</sub>, D<sub>2</sub>, C<sub>3</sub>) simultaneously

### Material Assumptions

1. **Raw material availability**: Sufficient raw materials (bricks) available at S<sub>1</sub>
2. **Material conservation**: No material loss except through defects/scrap
3. **Fixed material requirements**: n<sub>b,i</sub> is constant per unit

### Processing Assumptions

1. **Worker independence**: Processing time scales inversely with number of workers
2. **Transport time**: Independent of worker count
3. **Shift constraints**: Processing only during working hours (if shifts are modeled)
4. **Service time distributions**: Known distributions (triangular, normal, etc.)

### Demand Assumptions

1. **Stochastic demand**: D follows a known probability distribution
2. **No backorders**: Unsold goods have zero salvage value (or modeled separately)
3. **Price exogeneity**: Price p is given (not a decision variable in this model)

### Inventory Assumptions

1. **Holding cost linearity**: Holding cost is linear in inventory level
2. **No stockouts at buffers**: Stages cannot start if input buffers are empty (starvation)
3. **Blocking**: Stages cannot complete if output buffers are full

---

## 11. Operational Objective

Given the profit maximization objective, the **operational objective** becomes:

minimize C = C<sub>material</sub> + C<sub>processing</sub> + C<sub>inventory</sub> + C<sub>other</sub>

subject to:
- Capacity constraints
- Material flow constraints
- Inventory constraints
- Quality constraints (defect rates)
- Demand satisfaction: Q ≥ E[D] (or service level constraint)

**Decision focus:**
- **Optimize Q**: Determine optimal production quantity
- **Minimize costs**: Through optimal staffing (w<sub>i</sub>), buffer sizing (cap<sub>b</sub>), routing (P<sub>routing</sub>), and quality improvements

---

## 12. Simulation Alignment (Current Implementation)

**What the current DES implements:**
- **Push, single-product flow** with unlimited demand by default; revenue = unit_price × finished (or capped by `demand_qty` if set)
- **Cost hooks implemented** in code:
  - Material cost per released order (`unit_material_cost`)
  - Labor cost = busy_time × team_size × rate (`labor_costs_per_team_sec`)
  - Inventory holding cost = ∫ inventory dt × rate (`holding_costs_per_buffer_sec`)
  - Revenue parameter (`unit_price`), optional demand cap (`demand_qty`)
- **Single-server with faster service**: service time = τ<sub>i</sub>/w<sub>i</sub> + Δ<sub>i</sub> + Z<sub>i</sub>; parallel servers are not modeled.
- **Routing**: deterministic (balancing) or probabilistic, matches model constraints.
- **Defects/rework**: q<sub>i</sub>, r(i) implemented.

**Not yet implemented (future extensions):**
- Explicit stochastic demand and lost-sales/backorder logic
- Setup times, changeovers, learning curves
- Multi-product BOM with differentiated material consumption
- Price as a decision variable

---

## 13. Connection to Lean Principles

### Waste Elimination → Cost Reduction

The mathematical model directly connects to Lean principles:

1. **Overproduction waste** → High inventory holding costs C<sub>inventory</sub>
   - **Lean solution**: Reduce Q, implement pull (CONWIP K)

2. **Waiting waste** → High WIP → High holding costs
   - **Lean solution**: Balance flow, reduce variability

3. **Transportation waste** → High Δ<sub>i</sub> → Higher processing costs
   - **Lean solution**: Reduce transport times, optimize layout

4. **Processing waste** → High τ<sub>i</sub> → Higher processing costs
   - **Lean solution**: Process improvement, worker training

5. **Defect waste** → High C<sub>defect</sub>
   - **Lean solution**: Quality improvement (reduce q<sub>i</sub>)

6. **Motion waste** → Inefficient worker allocation
   - **Lean solution**: Optimize w<sub>i</sub> allocation

7. **Inventory waste** → High C<sub>inventory</sub>
   - **Lean solution**: Reduce buffer capacities, implement JIT

### Lean Transition

As the system transitions to Lean:
- **Costs decrease** through waste elimination
- **Profit increases** for the same Q (or higher Q achievable with same resources)
- **Decision variables** (w<sub>i</sub>, cap<sub>b</sub>, routing) are optimized to minimize waste

---

## 14. Key Performance Indicators (KPIs)

### Throughput
θ = lim<sub>t→∞</sub> C(t) / t

### Average Lead Time (Little's Law)
W̄ = L̄ / θ

where:
- **L̄** = Average WIP
- **W̄** = Average lead time

### Average WIP
L̄ = (1/T) ∫₀^T L(t) dt

### Utilization
ρ<sub>i</sub> = U<sub>i</sub>(T) / T

### Service Level
SL = C(T) / Q<sub>target</sub>

### Defect Rate
DR<sub>i</sub> = Q<sub>i,defect</sub> / Q<sub>i</sub>

---

## 15. Model Calibration

### Parameter Estimation

| Model Parameter | Source | Calibration Method |
|----------------|--------|-------------------|
| **p** | Market data | Market price per glider |
| **c<sub>b</sub>** | Material cost | Cost per brick |
| **c<sub>l,i</sub>** | Labor cost | Wage rate per worker per unit time |
| **h<sub>b</sub>** | Inventory cost | Holding cost rate × unit value |
| **τ<sub>i</sub>** | Time studies | Base processing time measurement |
| **q<sub>i</sub>** | Quality data | Historical defect rates |
| **π<sub>miss</sub>** | Historical data | Disruption frequency |
| **D** | Demand forecast | Demand distribution estimation |

### Model Validation

- Compare model predictions with actual production data
- Validate cost estimates against accounting records
- Verify capacity constraints match physical limitations
- Check material flow conservation in practice

---

*Document Version: 1.0*  
*Last Updated: 2024*

