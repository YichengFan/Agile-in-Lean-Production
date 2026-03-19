# Mathematical Model for LEGO Lean Production System

## 1. Objective Function

### Primary Objective: Maximize Profit

**Maximize profit:** $$\Pi = R - C$$

where:
- $R$ = Revenue
- $C$ = Total Cost

### Revenue Model (Mixed-Model)

Unlike a model with variant-specific pricing, the current system utilizes a global unit price for all variants to simplify financial tracking.

$$R = p \cdot \sum_{v \in V} \min(Q_v, D_v)$$

where:
- $V$ = Set of product models (e.g., m01, m02, m03, m04)
- $p$ = Global selling price per finished glider (€/unit)
- $Q_v$ = Produced finished goods quantity for model $v$ (units)
- $D_v$ = Random demand for model $v$ (units)

**Assumptions:**
- Unsold finished goods have zero salvage value (penalized as dead stock).
- All produced units that meet demand are sold at the global price $p$.

---

## 2. Cost Components

### Total Cost Decomposition

$$C = C_{material} + C_{processing} + C_{inventory} + C_{other}$$

where:
- $C_{material}$ = Material costs (bricks)
- $C_{processing}$ = Processing costs (labor paid per shift/simulation time)
- $C_{inventory}$ = Inventory holding costs (WIP, Raw Waste, Dead Stock)
- $C_{other}$ = Other costs (currently treated as 0 in explicit accounting, see Section 6)

---

## 3. Material Costs

### Material Cost Model

The accounting of material costs utilizes a global unit material cost ($c_m$) and depends entirely on the production strategy (Push vs. Pull):

$$C_{material} = \begin{cases} c_m \cdot Q_{procured} & \text{if Push mode (Forecast-driven)} \\ c_m \cdot Q_{started} & \text{if Pull mode (CONWIP-driven)} \end{cases}$$

where:
- $c_m$ = Global cost per unit material set (€/unit)
- $Q_{procured}$ = Total quantity procured based on forecast + waste rate
- $Q_{started}$ = Total quantity of orders released into the system

### Brick Requirements (Variant-Specific BOM)

For operational tracking and bottleneck analysis:

$$N_{b}(Q) = \sum_{v \in V} \sum_{i=1}^{m} \sum_{k \in \mathcal{K}} m_{i,k,v} \cdot Q_{i,v}$$

where:
- $m_{i,k,v}$ = Quantity of item/brick $k$ consumed per unit of model $v$ at stage $i$ (BOM)
- $Q_{i,v}$ = Quantity of model $v$ processed at stage $i$
- $\mathcal{K}$ = Set of item types

---

## 4. Processing Costs (Labor)

### Processing Cost Model

Labor costs are paid based on the total simulation time (e.g., shifts), regardless of whether workers are actively processing or idle due to starvation/blocking.

$$C_{processing} = \sum_{i=1}^{m} ( c_{l,i} \cdot w_i \cdot T )$$

where:
- $c_{l,i}$ = Labor cost per worker per unit time at stage $i$ (€/worker·min)
- $w_i$ = Number of workers at stage $i$ (decision variable)
- $T$ = Total simulation time (min)

### Service Time Model (Deterministic Diminishing Returns)

Worker efficiency does not scale linearly. Coordination overhead creates diminishing returns modeled via a square root function. Processing times are deterministic; variance is introduced primarily through defect rework loops.

$$S_{i,j} = \frac{\tau_i}{\sqrt{w_i}} + \Delta_i$$

where:
- $S_{i,j}$ = Processing time for unit $j$ at stage $i$ (min)
- $\tau_i$ = Base processing time at stage $i$ (min)
- $\sqrt{w_i}$ = Efficiency scaling factor based on team size
- $\Delta_i$ = Transport time at stage $i$ (min, unaffected by workers)

---

## 5. Inventory Holding Costs

To reflect Lean realities, inventory costs are decomposed into healthy WIP and wasteful dead stock.

### Total Inventory Cost

$$C_{inventory} = C_{WIP} + C_{raw\_waste} + C_{dead\_stock}$$

#### 1. WIP Holding Cost (Line Operations)
$$C_{WIP} = \sum_{b \in B \setminus \{A, E\}} h_b \cdot \bar{X}_b(T) \cdot T$$
- Excludes Raw Material (A) and Finished Goods (E) buffers.
- $h_b$ = Holding cost rate per minute at buffer $b$.

#### 2. Raw Material Waste (Push Mode Only)
$$C_{raw\_waste} = \max(0, Q_{procured} - Q_{sales}) \cdot T \cdot h_A$$
- Penalizes procurement that does not convert to actual sales.

#### 3. Finished Goods Dead Stock (Overproduction Waste)
$$C_{dead\_stock} = \max(0, Q_{produced} - Q_{sales}) \cdot \left(\frac{T}{2}\right) \cdot h_E$$
- Penalizes overproduction. Assumes unsold goods sit in the warehouse for an average of half the simulation time.

### Average Inventory Level

$$\bar{X}_b(T) = \frac{1}{T} \int_0^T X_b(t) dt$$

---

## 6. Other Costs (Diagnostics & Quality)

### Defect and Quality Costs (Implicit)

### Financial Diagnostics

These metrics are calculated for diagnostic purposes and are not added to $C$ to avoid double-counting.

**1. Opportunity Loss (Unmet Demand):**

$$C_{opp\_loss} = margin \cdot \sum_{v \in V} \max(0, D_v - Q_{sales,v})$$

- Represents profit left on the table due to unmet demand. $margin = p - c_m$.

**2. Overproduction Waste Cost:**

$$C_{\text{over waste}} = (c_m + \text{avg labor per unit}) \cdot \sum_{v \in V} \max(0, Q_{\text{produced},v} - D_v)$$

---

## 7. Decision Variables

### Primary Decision Variables

1. **$Q$** = Production quantity (finished gliders) - **primary decision**
2. **$w_i$** = Number of workers at stage $i$, $i \in \{1, 2, ..., m\}$
3. **$cap_b$** = Buffer capacity (Kanban limits) at buffer $b$
4. **$K$** = CONWIP level (if using pull strategy)

### Secondary Decision Variables

- **$\lambda$** = Release rate (for push strategy)

---

## 8. Constraints

### Capacity Constraints

**Stage capacity:** $$\theta_i \le \mu_i(w_i) = \frac{1}{\frac{\tau_i}{\sqrt{w_i}} + \Delta_i}$$

**Bottleneck constraint:** $\theta = \min_i \theta_i$

### Material Flow Constraints

**Conservation of flow (Input consumption):**
$$\lambda_{b, out} = m_{b, i} \cdot \theta_i \quad \forall b \in inputs(i)$$

**Conservation of flow (Output generation):**
$$\lambda_{b, in} = \theta_i \quad \forall b \in outputs(i)$$

**Multi-input synchronization (for S5):**
$$\frac{\lambda_{D1, out}}{m_{D1, 5}} = \frac{\lambda_{D2, out}}{m_{D2, 5}} = \frac{\lambda_{C3, out}}{m_{C3, 5}} = \theta_5$$
### Inventory & Pull Constraints

**Buffer capacity:** $0 \le X_b(t) \le cap_b$ for all $t$  
**CONWIP Limit:** $L(t) \le K$. If $L(t) = K$, new orders enter $L_{backlog}(t)$.

---

## 9. State Variables and Transition Functions

### State Variables

**Buffer states:**
- **$X_b(t)$** = Inventory level at buffer $b$ at time $t$

**Stage states:**
- **$Y_i(t)$** $\in \{0, 1\}$ = Busy status of stage $i$ (0 = idle, 1 = busy)
- **$U_i(t)$** = Cumulative busy time of stage $i$ up to time $t$

**System states:**
- **$L(t)$** = Work-in-process (WIP) at time $t$
- **$L_{backlog}(t)$** = Order backlog waiting for CONWIP clearance
- **$C(t)$** = Cumulative finished goods count up to time $t$
- **$Q(t)$** = Cumulative production quantity up to time $t$

### State Transition Functions

**Release event (t = $t_{release}$):** If $L(t^-) < K$: $L(t^+) = L(t^-) + 1$; $Q_{started}(t^+) = Q_{started}(t^-) + 1$  
If $L(t^-) \ge K$: $L_{backlog}(t^+) = L_{backlog}(t^-) + 1$

**Stage start (i at time t):** $Y_i(t^+) = 1$; $X_{b,in}(t^+) = X_{b,in}(t^-) - 1$ for all $b \in inputs(i)$

**Stage completion (i at time t):** $Y_i(t^+) = 0$; $U_i(t^+) = U_i(t^-) + S_{i,j}$

**Defect with prob $q_i$:** If rework $\rightarrow$ send to $r(i)$; if scrap $\rightarrow$ $L(t^+) = L(t^-) - 1$

**Finished goods (S5 $\rightarrow$ E):** $C(t^+) = C(t^-) + 1$; $L(t^+) = L(t^-) - 1$; $Q(t^+) = Q(t^-) + 1$  
*If auto-release enabled and $L_{backlog}(t^-) > 0$, trigger new Release event.*

---

## 10. Model Assumptions

### Production System Assumptions

1. **Mixed-Model Production**: The system supports multiple product variants ($v \in V$) flowing simultaneously, each requiring variant-specific BOMs.
2. **Pricing**: Unit price ($p$) and unit material cost ($c_m$) are treated as global constants across all product variants.
3. **Multi-input synchronization**: S5 requires all inputs (D1, D2, C3) simultaneously.

### Processing Assumptions

1. **Deterministic Processing**: Processing times are deterministic based on base time and worker efficiency ($\sqrt{w_i}$). 
2. **Variability**: Variability is introduced primarily through defect rework loops and stochastic demand noise (in Push mode).
3. **Time Unit**: All temporal logic is calculated in **minutes**.

### Demand Assumptions

1. **Stochastic demand (Push)**: Demand follows a known probability distribution with forecast/realization noise.
2. **No backorders**: Unmet demand results in opportunity loss; it cannot be backordered.

---

## 11. Operational Objective

Given the profit maximization objective, the **operational objective** becomes:

minimize $$C = C_{material} + C_{processing} + C_{inventory} + C_{other}$$

subject to:
- Capacity constraints
- Material flow constraints
- Inventory constraints
- Demand satisfaction: $Q \ge E[D]$ (or service level constraint)

**Decision focus:**
- **Optimize Q**: Determine optimal production quantity.
- **Minimize costs**: Through optimal staffing ($w_i$), buffer sizing ($cap_b$), and pull control ($K$).

---

## 12. Simulation Alignment (Current Implementation)

**What the current DES implements:**
- **Multi-product Mixed-flow**: Supports m01-m04 with variant-specific BOMs and routing.
- **Pull Control**: Fully implements CONWIP gating, Backlog queues, and Kanban buffer limits.
- **Cost Hooks**: 
  - Labor costs paid across full simulation time ($T$).
  - $C_{inventory}$ decomposed into WIP, Raw Waste, and Dead Stock.
  - Opportunity Loss and Overproduction Waste calculated for diagnostic KPIs.
- **Deterministic Non-linear Processing**: Implements $\tau_i/\sqrt{w_i}$ efficiency scaling.

---

## 13. Connection to Lean Principles

The mathematical model directly connects to Lean principles:

1. **Overproduction waste** $\rightarrow$ High dead stock inventory costs ($C_{dead\_stock}$)
   - **Lean solution**: Reduce Q, implement pull (CONWIP K).
2. **Waiting waste** $\rightarrow$ High WIP $\rightarrow$ High WIP holding costs ($C_{WIP}$)
   - **Lean solution**: Balance flow, reduce variability, implement Kanban ($cap_b$).
3. **Processing waste** $\rightarrow$ High $\tau_i$ $\rightarrow$ Higher processing costs
   - **Lean solution**: Process improvement, optimal worker allocation ($w_i$).
4. **Defect waste** $\rightarrow$ Scrapped WIP and wasted $T$
   - **Lean solution**: Quality improvement.

---

## 14. Key Performance Indicators (KPIs)

### System Flow
- **Throughput**: $\theta = \lim_{t \to \infty} C(t) / t$
- **Average Lead Time (Little's Law)**: $\bar{W} = \bar{L} / \theta$
- **Average Cycle Time**: Mean time between consecutive units entering buffer E.
- **Average WIP**: $\bar{L} = (1/T) \int_0^T L(t) dt$

### Efficiency & Quality
- **Utilization**: $\rho_i = U_i(T) / T$
- **Service Level**: $SL = Q_{sales} / D_{total}$
- **Defect Rate**: $DR_i = Q_{i,defect} / Q_i$

---

## 15. Model Calibration

### Parameter Estimation

| Model Parameter | Source | Calibration Method |
|----------------|--------|-------------------|
| **$p$** | Market data | Global market price |
| **$c_m$** | Material cost | Global material cost per unit |
| **$c_{l,i}$** | Labor cost | Wage rate per worker per minute |
| **$h_b$** | Inventory cost | Holding cost rate (differentiated by WIP/FG) |
| **$\tau_i$** | Time studies | Base processing time measurement |
| **$q_i$** | Quality data | Historical defect rates |
| **$D_v$** | Demand forecast | Demand distribution estimation |

---
*Document Version: 3.1* 
