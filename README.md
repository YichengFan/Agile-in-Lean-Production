# LEGO Lean Production Simulator

Discrete-event simulation (DES) for a lean LEGO production line with **Push** (demand-forecast) and **Pull** (CONWIP + Kanban) modes and **multi-model production** (four product variants).

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or run the CLI example: `python env.py`

## Documentation (where to find what)

| Document | Purpose | When to read |
|----------|---------|----------------|
| **[simulation_overview.md](simulation_overview.md)** | **Simulation model specification** — system overview, Push/Pull modes, event logic, parameters, KPIs, configuration, implementation mapping | Start here for how the simulator works and how it is configured. |
| [MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md) | **Multi-model production** — model definitions, BOM configuration per stage, demand forecasting with multiple models, usage examples, troubleshooting | Use when configuring or using multiple product models (m01–m04) or integrating forecast with production. |
| [mathematical_model.md](mathematical_model.md) | **Mathematical model** — objective function, decision variables, constraints, cost formulations | Use for the optimization/theoretical formulation behind the simulation. |

## Repo layout

| File | Role |
|------|------|
| **env.py** | DES engine and default `CONFIG` (stages, buffers, parameters) |
| **app.py** | Streamlit UI (Push/Pull mode, Order Release, CONWIP/Kanban, parameters) |
| **simulation_overview.md** | Authoritative simulation documentation (parameters, logic, KPIs) |

---

*For full simulation details (parameters, event logic, KPIs, file structure), see [simulation_overview](simulation_overview). For multi-model configuration and usage, see [MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md).*
