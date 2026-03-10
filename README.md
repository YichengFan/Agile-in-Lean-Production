# LEGO Lean Production Simulator

Discrete-event simulation (DES) for a lean LEGO production line with **Push** (demand-forecast) and **Pull** (CONWIP + Kanban) modes and multi-model production.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or run the CLI example: `python env.py`

## Documentation

| Document | Description |
|----------|-------------|
| **[readme3.md](readme3.md)** | **Simulation model (current)** — system overview, Push/Pull modes, parameters, defect/rework, event logic, KPIs, configuration |
| [mathematical_model.md](mathematical_model.md) | Mathematical model (objective, constraints, cost formulations) |
| [MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md) | Multi-model production guide |

## Repo layout

- **env.py** — DES engine and default `CONFIG`
- **app.py** — Streamlit UI (mode, Order Release, CONWIP/Kanban, parameters)
- **readme3.md** — Authoritative simulation documentation

---

*For full simulation details (parameters, event logic, KPIs, file structure), see [readme3.md](readme3.md).*
