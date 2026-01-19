# env.py
# LEGO Lean Production Simulation Environment (Discrete-Event)
# --------------------------------------------------------------------------------------
# This is a self-contained, dependency-light discrete-event simulation (DES) environment
# for a lean-production LEGO line. It encodes BOTH the simulation engine and a default
# environment configuration directly in this file — no Excel/JSON loading required.
#
# Key features
# - Discrete-event engine using a priority queue (heapq)
# - Entities: Buffer (inventory), Team (worker group), Stage (process node)
# - Multi-input Stage support (e.g., Final Assembly needs D1 + D2 + C3)
# - Single-output buffer OR probabilistic output routing (e.g., Set Sorting -> C1/C2/C3)
# - Shift schedules (optional), random disruptions, defects & rework
# - Time distributions: constant, normal, lognormal, triangular, uniform, exponential
# - KPIs: throughput/sec, average lead time, average WIP, team utilization, finished count
# - “Click to play”: step(), run_for(dt), run_until(t_stop) and a minimal example at bottom
#
# IMPORTANT
# - All names/IDs are in English for consistency.
# - Tune the CONFIG dict below to match your actual parameters.
# - Keep units consistent. This file uses **minutes** for simulated time.
# --------------------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import heapq
import random
import math
from collections import deque  # Used to track order release times for lead-time calculation



# ==============================================================================
# Utility helpers
# ==============================================================================

def sample_time(dist: Dict[str, Any], base: float) -> float:
    """
    Sample a processing time given a distribution descriptor.
    Supported dist['type']: 'constant', 'normal', 'lognormal', 'triangular',
                            'uniform', 'exponential'.
    Fallback is 'constant' with the given base time.
    """
    if not dist:
        return max(0.0, float(base))
    t = (dist.get("type") or "constant").lower()
    p1, p2, p3 = dist.get("p1"), dist.get("p2"), dist.get("p3")

    if t == "constant":
        return max(0.0, float(base))

    if t == "normal":
        mu = float(p1) if p1 is not None else float(base)
        sigma = float(p2) if p2 is not None else max(1e-9, 0.1 * mu)
        return max(0.0, random.gauss(mu, sigma))

    if t == "lognormal":
        mu = float(p1) if p1 is not None else math.log(max(1e-6, base))
        sigma = float(p2) if p2 is not None else 0.25
        return max(0.0, random.lognormvariate(mu, sigma))

    if t == "triangular":
        low = float(p1) if p1 is not None else 0.5 * base
        mode = float(p2) if p2 is not None else base
        high = float(p3) if p3 is not None else 1.5 * base
        return max(0.0, random.triangular(low, high, mode))

    if t == "uniform":
        a = float(p1) if p1 is not None else 0.8 * base
        b = float(p2) if p2 is not None else 1.2 * base
        return max(0.0, random.uniform(a, b))

    if t == "exponential":
        lam = 1.0 / float(base) if base else 1.0
        return max(0.0, random.expovariate(lam))

    # Fallback to constant
    return max(0.0, float(base))


# ==============================================================================
# Data structures
# ==============================================================================

@dataclass
class Buffer:
    """Finite (or infinite if capacity=None) storage for parts/products in itemized mode."""
    buffer_id: Any
    name: str
    capacity: Optional[int] = None
    initial_stock: Dict[str, int] = field(default_factory=dict)

    # internal state
    items: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        # itemized inventory only
        if isinstance(self.initial_stock, dict):
            self.items = {str(k): int(v) for k, v in self.initial_stock.items()}
        else:
            # if given as scalar, treat as generic items
            qty = int(self.initial_stock or 0)
            self.items = {f"{self.buffer_id}_item": qty} if qty > 0 else {}

    def can_pull_item(self, item_id: str, qty: int = 1) -> bool:
        return self.items.get(str(item_id), 0) >= qty

    def pull_item(self, item_id: str, qty: int = 1) -> Optional[Dict[str, int]]:
        key = str(item_id)
        if self.can_pull_item(key, qty):
            self.items[key] -= qty
            if self.items[key] <= 0:
                self.items.pop(key, None)
            return {key: qty}
        return None

    def can_push_item(self, item_id: str, qty: int = 1) -> bool:
        if self.capacity is None:
            return True
        return (self.total_items() + qty) <= int(self.capacity)

    def push_item(self, item_id: str, qty: int = 1) -> bool:
        if not self.can_push_item(item_id, qty):
            return False
        key = str(item_id)
        self.items[key] = self.items.get(key, 0) + qty
        return True

    def total_items(self) -> int:
        return sum(self.items.values())


@dataclass
class Team:
    """Worker group that can be busy/idle. Size used for visibility; one job at a time per Stage."""
    team_id: Any
    name: str
    size: int = 1
    shift_id: Optional[Any] = None

    # utilization tracking
    busy_time: float = 0.0          # accumulated busy time (simulation time units)
    last_busy_start: Optional[float] = None

    def start_busy(self, t: float):
        if self.last_busy_start is None:
            self.last_busy_start = t

    def stop_busy(self, t: float):
        if self.last_busy_start is not None:
            self.busy_time += max(0.0, t - self.last_busy_start)
            self.last_busy_start = None


@dataclass
class Stage:
    """
    Process node (BOM-driven, deterministic outputs).
    - Pulls required_materials (item_id -> qty) across input buffers.
    - Pushes deterministic outputs: output_buffers (buffer_id -> {item_id: qty}).
    - workers: number of workers at this stage. Processing time is reduced by sqrt(workers).
    - Supports model-specific materials: required_materials_by_model (model_id -> {item_id: qty})
    
    Time Calculation:
    - base_process_time_min: Base time to complete this stage's work in minutes (user-defined, independent of material count)
    - Worker efficiency: effective_time = base_time / sqrt(workers) (deterministic, models diminishing returns)
    - Distribution: ptime ~ Distribution(effective_time) (adds variability)
    - Transport: final_time = ptime + transport_time_min
    """
    stage_id: Any
    name: str
    team_id: Any
    input_buffers: List[Any] = field(default_factory=list)  # e.g., ['D1', 'D2', 'C3'] for Final Assembly
    required_materials: Dict[str, int] = field(default_factory=dict)  # BOM-style requirements (item_id -> qty) - legacy/default
    required_materials_by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)  # model_id -> {item_id: qty}
    output_buffers: Dict[str, Dict[str, int]] = field(default_factory=dict)  # deterministic outputs with items
    output_buffers_by_model: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)  # model_id -> {buffer_id: {item_id: qty}}

    base_process_time_min: float = 1.0
    time_distribution: Dict[str, Any] = field(default_factory=dict)
    transport_time_min: float = 0.0
    #dic to save the time from s2 to c1,c2,c3
    transport_time_to_outputs_min: dict = field(default_factory=dict)
    defect_rate: float = 0.0
    rework_stage_id: Optional[Any] = None
    workers: int = 1                                        # number of workers (affects processing speed)

    # internal state
    busy: bool = False


@dataclass(order=True)
class Event:
    """
    Timed event for the DES engine.
    - time: event time (simulation time)
    - seq: tie-breaker sequence number
    - kind: event name ('try_start', 'complete', etc.)
    - payload: dictionary with contextual data
    """
    time: float
    seq: int
    kind: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)


# ==============================================================================
# Simulation environment
# ==============================================================================

class LegoLeanEnv:
    """Discrete-event simulation environment for the LEGO lean production flow."""

    def __init__(self, config: Dict[str, Any], time_unit: str = "min", seed: Optional[int] = None):
        self.cfg = config
        self.time_unit = time_unit
        # Trace log
        self.t: float = 0.0
        self.log: List[str] = []
        if seed is not None:
            random.seed(seed)

        # Build buffers
        self.buffers: Dict[Any, Buffer] = {}
        for b in self.cfg.get("buffers", []):
            b_id = b.get("buffer_id") or b.get("name")
            initial_stock = b.get("initial_stock") or {}
            self.buffers[b_id] = Buffer(
                buffer_id=b_id,
                name=b.get("name") or str(b_id),
                capacity=b.get("capacity"),
                initial_stock=initial_stock
            )
        
        # Build teams
        self.teams: Dict[Any, Team] = {}
        for t in self.cfg.get("teams", []):
            t_id = t.get("team_id") or t.get("name")
            self.teams[t_id] = Team(
                team_id=t_id,
                name=t.get("name") or str(t_id),
                size=int(t.get("size") or 1),
                shift_id=t.get("shift_id")
            )

        # Build stages
        self.stages: Dict[Any, Stage] = {}
        for s in self.cfg.get("stages", []):
            s_id = s.get("stage_id") or s.get("name")
            # Normalize input buffers to list
            in_bufs = s.get("input_buffers")
            if isinstance(in_bufs, str) and in_bufs.strip():
                in_bufs = [in_bufs]
            in_bufs = in_bufs or []
            stage_team_id = s.get("team_id")
            self.stages[s_id] = Stage(
                stage_id=s_id,
                name=s.get("name") or str(s_id),
                team_id=stage_team_id,
                input_buffers=in_bufs,
                required_materials=s.get("required_materials") or {},
                required_materials_by_model=s.get("required_materials_by_model") or {},
                output_buffers=s.get("output_buffers") or {},
                output_buffers_by_model=s.get("output_buffers_by_model") or {},
                base_process_time_min=float(s.get("base_process_time_min") or s.get("base_process_time_sec", 1.0) / 60.0),  # Support both old and new names, convert sec to min
                time_distribution=s.get("time_distribution") or {},
                transport_time_min=float(s.get("transport_time_min") or s.get("transport_time_sec", 0.0) / 60.0),  # Support both old and new names, convert sec to min
                transport_time_to_outputs_min={k: float(v) / 60.0 if isinstance(v, (int, float)) else v for k, v in (s.get("transport_time_to_outputs_min") or s.get("transport_time_to_outputs_sec", {})).items()},
                defect_rate=float(s.get("defect_rate") or 0.0),
                rework_stage_id=s.get("rework_stage_id"),
                workers=int(s.get("workers") or 1)
            )

        # Global parameters
        self.parameters = self.cfg.get("parameters", {})
        self.finished_buffers = list(self.parameters.get("finished_buffer_ids", ["E"]))
        self.trace_assembly: bool = bool(self.parameters.get("trace_assembly", False))
        # Cost & revenue parameters (optional)
        cost_cfg = self.parameters.get("cost", {})
        self.unit_price = float(cost_cfg.get("unit_price", 0.0))
        self.unit_material_cost = float(cost_cfg.get("unit_material_cost", 0.0))

        # Material cost accounting mode
        self.material_cost_mode = (self.parameters.get("material_cost_mode") or "procure_forecast")

        # ----------------------------------------------------------------------
        # Margin definition (unit contribution margin for opportunity cost)
        # ----------------------------------------------------------------------
        # We define margin as a per-unit contribution margin used to monetize unmet demand:
        #   margin = unit_price - unit_material_cost
        # Labor and inventory holding are modeled separately in cost_total, so they MUST NOT be embedded here,
        # otherwise you'll double count them when computing profit.
        margin_cfg = cost_cfg.get("margin_per_unit", None)
        if margin_cfg is None:
            self.margin_per_unit = max(0.0, self.unit_price - self.unit_material_cost)
            self.log.append(
                f"{self._fmt_t()} Margin per unit defaulted to unit_price - unit_material_cost "
                f"= {self.unit_price:.4f} - {self.unit_material_cost:.4f} = {self.margin_per_unit:.4f}."
            )
        else:
            self.margin_per_unit = float(margin_cfg)
            self.log.append(
                f"{self._fmt_t()} Margin per unit overridden by config: {self.margin_per_unit:.4f}."
            )

        self.labor_costs_per_team = {
            str(k): float(v) for k, v in (cost_cfg.get("labor_costs_per_team_sec") or {}).items()
        }
        self.holding_costs_per_buffer = {
            str(k): float(v) for k, v in (cost_cfg.get("holding_costs_per_buffer_sec") or {}).items()
        }
        self.demand_qty = cost_cfg.get("demand_qty")
        if self.demand_qty is not None:
            try:
                self.demand_qty = int(self.demand_qty)
            except Exception:
                self.demand_qty = None
        self.revenue_total: float = 0.0
        self.cost_material: float = 0.0
        self.cost_labor: float = 0.0
        self.cost_inventory: float = 0.0
        self.cost_other: float = 0.0
        self.push_demand_enabled: bool = bool(self.parameters.get("push_demand_enabled", False))
        # In push mode, auto-release is always enabled (forecast automatically triggers production)
        self.push_auto_release: bool = True if self.push_demand_enabled else bool(self.parameters.get("push_auto_release", False))
        self.push_demand_horizon_weeks: int = int(self.parameters.get("push_demand_horizon_weeks", 3) or 3)
        self.push_weekly_demand_mean: float = float(self.parameters.get("push_weekly_demand_mean", 30.0))
        self.push_forecast_noise_pct: float = float(self.parameters.get("push_forecast_noise_pct", 0.1))
        self.push_realization_noise_pct: float = float(self.parameters.get("push_realization_noise_pct", 0.05))
        self.push_procurement_waste_rate: float = float(self.parameters.get("push_procurement_waste_rate", 0.05))
        self.supplier_stage_ids: set = set()
        # Demand bookkeeping (per-model)
        self.demand_forecast: Dict[str, List[int]] = {}  # model_id -> list of weekly forecasts
        self.demand_realized: Dict[str, List[int]] = {}  # model_id -> list of weekly realizations
        self.planned_release_qty: Dict[str, int] = {}  # model_id -> total planned qty
        self.realized_demand_total: Dict[str, int] = {}  # model_id -> total realized qty
        self.procured_qty: int = 0
        # Model probability distribution for demand allocation
        self.model_demand_probabilities: Dict[str, float] = {
            "m01": 0.60,  # Sly_Slider 60%
            "m03": 0.20,  # Icky_Ice_Glider 20%
            "m02": 0.10,  # Gliderlinski 10%
            "m04": 0.10   # Icomat_2000X 10%
        }
        # Override with config if provided
        if "model_demand_probabilities" in self.parameters:
            self.model_demand_probabilities.update(self.parameters["model_demand_probabilities"])
        # Scheduled releases tracking
        self._scheduled_releases: List[Dict[str, Any]] = []  # List of {time, model_id, qty}
        self.buffer_time_area: Dict[Any, float] = {b_id: 0.0 for b_id in self.buffers}
        self.last_buffer_time: float = 0.0
        # Event queue and time
        self._evt_seq = 0
        self._queue: List[Event] = []
        self.t: float = 0.0
        #每次S2完工产生一个job_id,并且记录还有几次deliver没送完
        # Per-output transport tracking (used when a stage schedules multiple deliveries, e.g. S2 -> C1/C2/C3)
        self._job_seq: int = 0
        self._pending_deliveries: Dict[int, Dict[str, Any]] = {}

        # Deduplicate try_start retries to avoid event explosion
        self._try_start_scheduled: set = set()


        # KPIs
        self.finished: int = 0
        self.finished_goods_by_model: Dict[str, int] = {}  # Track finished units per model
        self.started: int = 0
        self.lead_times: List[float] = []
        self.wip_time_area: float = 0.0
        self.last_wip_time: float = 0.0
        self.current_wip: int = 0
        # --------------------------------------------------------------------------
        # Release control: CONWIP + Kanban (recommended for merge-assembly lines)
        # --------------------------------------------------------------------------

        # Which stages act as "release stages" (where orders are injected into the system).
        # Even if a release stage has input buffers (e.g., S1 pulls from Warehouse A),
        # we still gate its starts via order tokens to implement CONWIP release control.
        param_release = self.parameters.get("release_stage_ids")
        if param_release is None:
            param_release = ["S1"]
        if isinstance(param_release, list) and len(param_release) == 0:
            param_release = ["S1"]
        self.release_stage_ids = set(param_release)
        self.log.append(f"Release stages detected: {self.release_stage_ids}")
        # Supplier stages should be empty by default (S1 processes materials normally)
        sup_ids = self.parameters.get("supplier_stage_ids", [])
        self.supplier_stage_ids = set(sup_ids if isinstance(sup_ids, list) else [sup_ids])


        # Global WIP cap for CONWIP; if None, CONWIP is disabled
        self.conwip_wip_cap = self.parameters.get("conwip_wip_cap", None)

        # If True, each finished unit automatically releases one new order (closed-loop CONWIP)
        self.auto_release_conwip = bool(self.parameters.get("auto_release_conwip", False))
        if self.push_demand_enabled:
            self.conwip_wip_cap = None
            self.auto_release_conwip = False

        # Kanban caps per buffer (local WIP control). These are "control limits",
        # different from physical capacities (which may be very large like 9999).
        self.kanban_caps = {str(k): int(v) for k, v in (self.parameters.get("kanban_caps", {}) or {}).items()}

        # Count how often a stage is prevented from starting due to Kanban cap
        self.kanban_blocking_counts = {s_id: 0 for s_id in self.stages}

        # Order tokens per release stage (how many jobs this release stage is allowed to start)
        self.stage_orders = {s_id: 0 for s_id in self.release_stage_ids}

        # FIFO queue of release timestamps (one per released unit) to compute lead times at completion
        self._release_times = deque()
        
        # Model type tracking: map job_id to model_id
        self._job_model_map: Dict[int, str] = {}
        
        # Model definitions from config
        self.model_definitions = self.cfg.get("models", {})  # model_id -> model config

        # Timeline sampling
        self.timeline: List[Dict[str, Any]] = []
        self._sample_dt: float = float(self.parameters.get("timeline_sample_dt_min", 5.0) or self.parameters.get("timeline_sample_dt_sec", 5.0) / 60.0 or 5.0)  # Support both old and new names
        self._next_sample_t: float = self._sample_dt
        self._last_sample_finished: int = 0
        self._last_sample_time: float = 0.0
        # Assembly traces (optional)
        self.assembly_traces: List[Dict[str, Any]] = []

        # Shift logic disabled - always start at t=0 for continuous production
        # (No shift alignment needed - production runs 24/7)

        # KPI counters per stage
        self.stage_completed_counts: Dict[Any, int] = {s_id: 0 for s_id in self.stages}
        self.starvation_counts: Dict[Any, int] = {s_id: 0 for s_id in self.stages}
        self.blocking_counts: Dict[Any, int] = {s_id: 0 for s_id in self.stages}
        self.stage_defect_counts: Dict[Any, int] = {s_id: 0 for s_id in self.stages}

        # Push demand plan (finite push based on forecast)
        if self.push_demand_enabled:
            self._init_push_demand_plan()

    # --------------------------------------------------------------------------
    # Shift logic
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Event queue helpers
    # --------------------------------------------------------------------------

    def _push_event(self, when: float, kind: str, payload: Optional[Dict[str, Any]] = None):
        """Schedule an event at time 'when'."""
        self._evt_seq += 1
        heapq.heappush(self._queue, Event(when, self._evt_seq, kind, payload or {}))

    def _pop_event(self) -> Optional[Event]:
        if not self._queue:
            return None
        return heapq.heappop(self._queue)


    def _schedule_try_start(self, stage_id: Any, delay: float = 0.0, is_rework: bool = False):
        """Schedule a try_start event if one isn't already pending for this (stage_id, is_rework)."""
        key = (stage_id, bool(is_rework))
        if key in self._try_start_scheduled:
            return
        self._try_start_scheduled.add(key)
        self._push_event(self.t + float(delay), "try_start", {"stage_id": stage_id, "is_rework": bool(is_rework)})

    def _apply_worker_efficiency(self, base_time: float, workers: int) -> float:
        """
        Apply deterministic worker efficiency function using square root.
        
        Formula: effective_time = base_time / sqrt(workers)
        
        This models realistic diminishing returns:
        - More workers reduce time, but not linearly
        - Coordination overhead and task dependencies limit perfect parallelization
        
        Args:
            base_time: Base processing time (user-defined per stage)
            workers: Number of workers at the stage
            
        Returns:
            Effective processing time after worker efficiency adjustment
        """
        workers = max(1, int(workers))  # Ensure at least 1 worker
        return base_time / math.sqrt(workers)

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def enqueue_orders(self, qty: int = 1, model_id: Optional[str] = None):
        """
        Release 'qty' orders into the system (CONWIP-aware).
        Adds order tokens to release stages (e.g., S1) and triggers try_start.
        
        Args:
            qty: Number of orders to release
            model_id: Model type identifier (e.g., "m01", "m02", "m03", "m04"). 
                     If None, uses default model or first available model.
        """
        qty = int(qty)
        if qty <= 0:
            return

        # Determine model_id if not provided
        if model_id is None:
            # Use default model from config, or first model if available
            model_id = self.parameters.get("default_model_id")
            if model_id is None and self.model_definitions:
                model_id = list(self.model_definitions.keys())[0]
            if model_id is None:
                model_id = "default"  # fallback to legacy behavior

        # Enforce CONWIP WIP cap (if enabled)
        cap = self.conwip_wip_cap
        if cap is not None:
            allowed = max(0, int(cap) - int(self.current_wip))
            qty = min(qty, allowed)
            if qty <= 0:
                self.log.append(f"{self._fmt_t()} CONWIP cap reached; no new orders released.")
                return

        # Update WIP area before changing WIP level
        self._accumulate_wip(self.t)
        self.current_wip += qty
        self.started += qty

        # Track release timestamps for lead-time KPI (FIFO)
        # Also track model types for each release
        for _ in range(qty):
            self._release_times.append(self.t)
            # We'll assign model_id to job_id when the job starts at the release stage

        # Allocate order tokens to release stages and trigger try_start
        # Store model_id in stage orders for later assignment to jobs
        for s_id in self.release_stage_ids:
            current_orders = self.stage_orders.get(s_id, 0)
            self.stage_orders[s_id] = current_orders + qty
            # Store model_id for this batch of orders (we'll use it when jobs start)
            if not hasattr(self, '_stage_order_models'):
                self._stage_order_models = {}
            if s_id not in self._stage_order_models:
                self._stage_order_models[s_id] = []
            # Append model_id for each order in this batch
            self._stage_order_models[s_id].extend([model_id] * qty)
            self._schedule_try_start(s_id, delay=0.0, is_rework=False)

        self.log.append(f"{self._fmt_t()} Released {qty} order(s) of model '{model_id}' into {sorted(self.release_stage_ids)}.")

    def step(self) -> Optional[Event]:
        """
        Process the next scheduled event and return it. Returns None if no events remain.
        """
        ev = self._pop_event()
        if ev is None:
            return None

        # Shift logic disabled - process all events immediately
        # (No shift windows - production runs continuously)

        # Advance simulation time
        self._on_time_advance(ev.time)

        # Dispatch to handler
        handler = getattr(self, f"_on_{ev.kind}", None)
        if handler:
            handler(ev)
        else:
            self.log.append(f"{self._fmt_t()} [WARN] No handler for event kind='{ev.kind}'.")
        return ev

    def run_for(self, dt: float, max_events: int = 100000):
        """Run the simulation for dt minutes from current time."""
        self.run_until(self.t + dt, max_events=max_events)

    def run_until(self, t_stop: float, max_events: int = 1000000):
        """Run the simulation until time reaches t_stop or event cap is hit."""
        count = 0
        while self._queue and count < max_events:
            if self._queue[0].time > t_stop:
                break
            self.step()
            count += 1
        # Only fast-forward if we naturally reached t_stop or queue is empty
        if count < max_events or not self._queue:
            self._on_time_advance(t_stop)

    def _on_unit_finished(self, qty: int = 1, model_id: Optional[str] = None):
        """Called when finished units enter a finished buffer (e.g., E).

        Updates:
          - finished count
          - finished_goods_by_model (per-model tracking)
          - lead times (FIFO match to release times)
          - WIP (decrement on completion)
          - optionally auto-releases CONWIP orders
        """
        qty = int(qty)
        if qty <= 0:
            return

        self.finished += qty
        
        # Track finished goods by model
        if model_id:
            self.finished_goods_by_model[model_id] = self.finished_goods_by_model.get(model_id, 0) + qty

        # Lead time = finish_time - release_time (FIFO matching)
        for _ in range(qty):
            rt = self._release_times.popleft() if self._release_times else 0.0
            self.lead_times.append(max(0.0, self.t - rt))

        # Unit leaves the system -> reduce WIP
        self._accumulate_wip(self.t)
        self.current_wip = max(0, self.current_wip - qty)

        # Closed-loop CONWIP (optional)
        if self.auto_release_conwip:
            self.enqueue_orders(qty)

    # --------------------------------------------------------------------------
    # KPI accumulation
    # --------------------------------------------------------------------------


    def _accumulate_wip(self, new_t: float):
        """Accumulate WIP*time area to compute average WIP later."""
        dt = max(0.0, new_t - self.last_wip_time)
        self.wip_time_area += self.current_wip * dt
        self.last_wip_time = new_t

    def get_kpis(self) -> Dict[str, Any]:
        """Return KPI snapshot for the elapsed simulation."""
        sim_time = max(1e-9, self.t)  # avoid div-by-zero
        throughput = self.finished / sim_time
        lead_time_avg = sum(self.lead_times) / len(self.lead_times) if self.lead_times else 0.0
        wip_avg = self.wip_time_area / sim_time
        service_level =( self.finished / self.started)if self.started > 0 else 0.0
        utilization = {}
        """Defect KPI calculation"""
        defect_rate_per_stage = {}
        total_defects = 0
        total_processed = 0

        for s_id in self.stages:
            defects = self.stage_defect_counts.get(s_id, 0)
            completed = self.stage_completed_counts.get(s_id, 0)
            total = defects + completed
            if total > 0:
                defect_rate_per_stage[s_id] = round(defects / total, 3)
            else:
                defect_rate_per_stage[s_id] = 0.0

            total_defects += defects
            total_processed += total
        """Overall defect rate (aggregated)"""

        total_defect_rate = round(total_defects / total_processed, 3) if total_processed > 0 else 0.0

        labor_cost = 0.0
        for team_id, team in self.teams.items():
            # If currently busy, close interval temporally for utilization calculation
            if team.last_busy_start is not None:
                team.stop_busy(self.t)
                team.start_busy(self.t)
            utilization[team_id] = team.busy_time / sim_time
            rate = self.labor_costs_per_team.get(str(team_id), 0.0)
            labor_cost += rate * team.size * team.busy_time

        # Inventory holding cost (area under inventory curves × holding rate)
        # Exclude buffer A from inventory cost calculation
        inventory_cost = 0.0
        for b_id, area in self.buffer_time_area.items():
            if str(b_id) == "A":
                continue  # Skip buffer A in inventory cost calculation
            h = self.holding_costs_per_buffer.get(str(b_id), 0.0)
            inventory_cost += h * area

        # Handle per-model forecasts (new structure) or legacy single forecast
        planned_release_qty_dict = getattr(self, "planned_release_qty", {})
        if isinstance(planned_release_qty_dict, dict):
            planned_release_qty = sum(planned_release_qty_dict.values())
        else:
            planned_release_qty = int(planned_release_qty_dict or 0)

        # demand_forecast_total equals planned_release_qty (removed redundant KPI)
        # demand_forecast_dict kept for per-model breakdown if needed
        demand_forecast_dict = getattr(self, "demand_forecast", {})

        realized_demand_dict = getattr(self, "realized_demand_total", {})
        if isinstance(realized_demand_dict, dict):
            # Per-model: sum all model realizations
            demand_realized_total = sum(realized_demand_dict.values())
        else:
            # Legacy: single value
            demand_realized_total = int(realized_demand_dict or 0)

        procured_qty = getattr(self, "procured_qty", 0)
        # ----------------------------
        # Demand → Sales → Opportunity cost (UNMET demand only)
        # ----------------------------
        # demand_total priority:
        # 1) push plan realized demand (if available)
        # 2) external demand cap demand_qty (for pull-mode experiments)
        # 3) None/0 => no unmet-demand penalty
        demand_total = 0
        if int(demand_realized_total) > 0:
            demand_total = int(demand_realized_total)
        elif self.demand_qty is not None:
            demand_total = int(self.demand_qty)

        # Sales units: cannot exceed demand_total if demand_total is defined
        if demand_total > 0:
            sales_units = min(int(self.finished), int(demand_total))
        else:
            sales_units = int(self.finished)

        # Revenue based on actual sales (avoid counting overproduction as sold)
        revenue_total = float(self.unit_price) * float(sales_units)

        cost_total = self.cost_material + labor_cost + inventory_cost + self.cost_other
        profit = revenue_total - cost_total

        # Opportunity cost (unmet demand only): m * (D - Q)^+
        unmet_demand = max(0, int(demand_total) - int(sales_units)) if demand_total > 0 else 0
        opportunity_cost = float(self.margin_per_unit) * float(unmet_demand)

        # Availability / service (demand-based)
        availability = (float(sales_units) / float(demand_total)) if demand_total > 0 else 1.0

        avg_buffer_levels = {
            str(b_id): (area / sim_time) if sim_time > 0 else 0.0
            for b_id, area in self.buffer_time_area.items()
        }
        # Persist latest cost/revenue snapshots for external inspection
        self.cost_labor = labor_cost
        self.cost_inventory = inventory_cost
        self.revenue_total = revenue_total
        return {
            "sim_time_min": sim_time,
            "throughput_per_min": throughput,
            "lead_time_avg_min": lead_time_avg,
            "wip_avg_units": wip_avg,
            "utilization_per_team": utilization,
            "finished_units": self.finished,
            "finished_goods_by_model": self.finished_goods_by_model.copy(),
            "planned_release_qty": planned_release_qty,
            "planned_release_qty_by_model": planned_release_qty_dict if isinstance(planned_release_qty_dict, dict) else {},
            "demand_realized_total": demand_realized_total,
            "demand_realized_by_model": realized_demand_dict if isinstance(realized_demand_dict, dict) else {},
            "procured_qty": procured_qty,
            "stage_completed_counts": self.stage_completed_counts,
            "starvation_counts": self.starvation_counts,
            "blocking_counts": self.blocking_counts,
            "service_level": service_level,
            "stage_defect_counts": self.stage_defect_counts,
            "defect_rate_per_stage": defect_rate_per_stage,
            "total_defect_rate": total_defect_rate,
            "revenue_total": revenue_total,
            "cost_material": self.cost_material,
            "cost_labor": labor_cost,
            "cost_inventory": inventory_cost,
            "cost_other": self.cost_other,
            "cost_total": cost_total,
            "profit": profit,
            "opportunity_cost": opportunity_cost,
            #2026-01-10
           # "profit_after_opportunity": profit_after_opportunity,
            "availability": availability,
            "avg_buffer_levels": avg_buffer_levels,
        }

    # --------------------------------------------------------------------------
    # Event handlers
    # --------------------------------------------------------------------------

    def _init_push_demand_plan(self):
        """
        Generate per-model forecast/realized demand for 3 weeks total.
        Allocates demand across models based on probability distribution.
        Schedules releases over time based on working hours (8h/day × 5 days/week × 3 weeks = 7200 min).
        
        Demand Generation Process:
        1. Generate total forecast for 3 weeks with forecast noise
        2. Allocate forecast across models based on probability distribution
        3. Apply realization noise to each model's forecast to get final demand
        4. Production is planned based on forecast (planned_release_qty)
        5. Actual demand is realized_demand_total (with noise applied)
        6. If realized_demand > planned_release, there will be unmet demand (push mode limitation)
        """
        horizon_weeks = max(1, int(self.push_demand_horizon_weeks))
        total_weekly_mean = max(0.0, float(self.push_weekly_demand_mean))
        f_noise = max(0.0, float(self.push_forecast_noise_pct))
        r_noise = max(0.0, float(self.push_realization_noise_pct))
        waste = max(0.0, float(self.push_procurement_waste_rate))

        # Get available model IDs
        available_models = list(self.model_definitions.keys()) if self.model_definitions else list(self.model_demand_probabilities.keys())
        if not available_models:
            available_models = ["m01"]  # Fallback to default model

        # Generate total forecast for 3 weeks (not per week, but total over 3 weeks)
        total_3week_mean = total_weekly_mean * horizon_weeks
        noise = random.uniform(-f_noise, f_noise)
        total_forecast = max(0, int(round(total_3week_mean * (1.0 + noise))))

        # Allocate total forecast across models based on probabilities
        model_forecasts: Dict[str, int] = {}
        remaining = total_forecast
        sorted_models = sorted(available_models, key=lambda m: self.model_demand_probabilities.get(m, 0.0), reverse=True)
        
        for i, model_id in enumerate(sorted_models):
            prob = self.model_demand_probabilities.get(model_id, 0.0)
            if i == len(sorted_models) - 1:
                # Last model gets remaining
                model_forecasts[model_id] = remaining
            else:
                qty = max(0, int(round(total_forecast * prob)))
                model_forecasts[model_id] = qty
                remaining -= qty

        # Generate per-model forecasts and realizations
        self.demand_forecast = {}
        self.demand_realized = {}
        self.planned_release_qty = {}
        self.realized_demand_total = {}

        total_planned = 0
        total_realized = 0

        for model_id, forecast_qty in model_forecasts.items():
            # Forecast for this model
            forecast_list = [forecast_qty]  # Single value for 3-week total
            self.demand_forecast[model_id] = forecast_list
            self.planned_release_qty[model_id] = forecast_qty
            total_planned += forecast_qty

            # Realization with noise (can be higher or lower than forecast)
            # This creates the scenario where push mode may not fulfill all demand.
            # 
            # Process:
            # 1. We plan production based on forecast (planned_release_qty = forecast_qty)
            # 2. Actual demand is forecast + realization noise (realized_qty)
            # 3. If realized_qty > forecast_qty, we have unmet demand (push mode limitation)
            # 4. If realized_qty < forecast_qty, we have overproduction
            #
            # Noise is applied symmetrically: can increase or decrease demand
            noise = random.uniform(-r_noise, r_noise)
            realized_qty = max(0, int(round(forecast_qty * (1.0 + noise))))
            realization_list = [realized_qty]
            self.demand_realized[model_id] = realization_list
            self.realized_demand_total[model_id] = realized_qty
            total_realized += realized_qty

        # Calculate procurement with waste
        self.procured_qty = int(math.ceil(total_planned * (1.0 + waste)))

        if self.material_cost_mode == "procure_forecast":
            self.cost_material = self.unit_material_cost * self.procured_qty

        # Queue orders immediately based on forecast (push mode)
        # When push_auto_release is enabled, enqueue all planned orders immediately
        # This triggers production to start right away based on the forecast
        if self.push_auto_release:
            # Queue orders immediately at simulation start for all planned quantities
            for model_id, planned_qty in self.planned_release_qty.items():
                if planned_qty > 0:
                    self.enqueue_orders(qty=planned_qty, model_id=model_id)
                    model_name = self.model_definitions.get(model_id, {}).get("name", model_id) if self.model_definitions else model_id
                    self.log.append(
                        f"{self._fmt_t()} [PUSH] Queued {planned_qty} order(s) of model '{model_name}' ({model_id}) "
                        f"based on demand forecast (planned_release_qty)."
                    )

        self.log.append(
            f"{self._fmt_t()} Push demand plan (3 weeks): "
            f"total_forecast={total_forecast}, total_realized={total_realized}, "
            f"per_model={model_forecasts}, procured={self.procured_qty}, waste_rate={waste:.3f}."
        )

    def _on_try_start(self, ev: Event):
        """Attempt to start a job at the stage, pulling inputs and engaging the team."""
        stage = self.stages.get(ev.payload.get("stage_id"))
        if stage is None:
            return

        is_rework = bool(ev.payload.get("is_rework", False))
        model_id = ev.payload.get("model_id")  # Get model_id from event if provided

        # This try_start is now being processed; clear any pending flag for this stage
        self._try_start_scheduled.discard((stage.stage_id, is_rework))
        # If stage is busy, do nothing. The stage will re-trigger try_start when it becomes free.
        #2026-1-3
        if stage.busy:
            return

        # Release-stage gating: release stages need an order token to start (except rework)
        if stage.stage_id in self.release_stage_ids and (not is_rework):
            if self.stage_orders.get(stage.stage_id, 0) <= 0:
                return
            self.stage_orders[stage.stage_id] -= 1
            # Get model_id from the queue for this release stage
            if hasattr(self, '_stage_order_models') and stage.stage_id in self._stage_order_models:
                if self._stage_order_models[stage.stage_id]:
                    model_id = self._stage_order_models[stage.stage_id].pop(0)
                else:
                    # Fallback if queue is empty
                    model_id = self.parameters.get("default_model_id") or "default"
            else:
                model_id = self.parameters.get("default_model_id") or "default"
        elif model_id is None:
            # For non-release stages, try to infer model_id from available materials
            # Check which model's BOM matches the available materials in input buffers
            inferred_model_id = None
            if stage.required_materials_by_model:
                # Try each model's BOM to see which one matches available materials
                for test_model_id, test_materials in stage.required_materials_by_model.items():
                    # Check if all required materials for this model are available
                    all_available = True
                    for item_id, required_qty in test_materials.items():
                        found = False
                        for b_id in stage.input_buffers:
                            buf = self.buffers.get(b_id)
                            if buf and buf.can_pull_item(item_id, required_qty):
                                found = True
                                break
                        if not found:
                            all_available = False
                            break
                    if all_available:
                        inferred_model_id = test_model_id
                        break
            
            if inferred_model_id:
                model_id = inferred_model_id
            else:
                # Fallback: try to infer from recent jobs or use default
                model_id = self.parameters.get("default_model_id") or "default"

        # Get model-specific materials if available, otherwise use default
        required_materials = stage.required_materials.copy() if stage.required_materials else {}
        if model_id and model_id != "default" and stage.required_materials_by_model:
            model_materials = stage.required_materials_by_model.get(model_id)
            if model_materials:
                required_materials = model_materials.copy()
        
        # Get model-specific outputs if available
        output_buffers = stage.output_buffers.copy() if stage.output_buffers else {}
        if model_id and model_id != "default" and stage.output_buffers_by_model:
            model_outputs = stage.output_buffers_by_model.get(model_id)
            if model_outputs:
                output_buffers = model_outputs.copy()

        # Kanban gating: prevent starting if controlled output buffers would exceed caps
        for out_b_id, materials in output_buffers.items():
            cap = self.kanban_caps.get(str(out_b_id))
            if cap is None:
                continue
            ob = self.buffers.get(out_b_id)
            if ob is None:
                continue

            projected = ob.total_items() + sum(int(q) for q in materials.values())
            if projected > cap:
                self.kanban_blocking_counts[stage.stage_id] += 1
                # Preserve model_id in retry
                self._push_event(self.t + 0.5, "try_start", {
                    "stage_id": stage.stage_id, 
                    "is_rework": is_rework,
                    "model_id": model_id
                })
                return

        # Check inputs (BOM mode only)
        pulled_items: List[tuple] = []  # (buf, item_id, qty) for rollback if needed

        bom_mode = bool(required_materials)

        if self.push_demand_enabled and stage.stage_id in self.supplier_stage_ids:
            all_push_success = True
            for out_buffer_id, materials in output_buffers.items():
                ob = self.buffers.get(out_buffer_id)
                if not ob:
                    self.log.append(f"{self._fmt_t()} '{stage.name}' error: missing output buffer '{out_buffer_id}'.")
                    all_push_success = False
                    continue
                for item_id, qty in materials.items():
                    if not ob.push_item(item_id, qty):
                        self.blocking_counts[stage.stage_id] += 1
                        self.log.append(f"{self._fmt_t()} '{stage.name}' output blocked: '{out_buffer_id}' full (supplier retry).")
                        self._push_event(self.t + 0.5, "try_start", {
                            "stage_id": stage.stage_id, 
                            "is_rework": is_rework,
                            "model_id": model_id
                        })
                        return
                    self.log.append(f"{self._fmt_t()} Supplier '{stage.name}' delivered {qty}x '{item_id}' → '{out_buffer_id}'.")
            if all_push_success:
                self.stage_completed_counts[stage.stage_id] += 1
                # Wake consumers of pushed buffers - pass model_id
                for s in self.stages.values():
                    for b in output_buffers.keys():
                        if b in s.input_buffers and not s.busy:
                            self._push_event(self.t, "try_start", {
                                "stage_id": s.stage_id, 
                                "is_rework": False,
                                "model_id": model_id
                            })
                # Try next supplier drop
                self._schedule_try_start(stage.stage_id, delay=0.0, is_rework=False)
            return

        if required_materials:
            for item_id, required_qty in required_materials.items():
                pulled = False
                for b_id in stage.input_buffers:
                    buf = self.buffers.get(b_id)
                    if buf and buf.pull_item(item_id, required_qty):
                        pulled_items.append((buf, item_id, required_qty))
                        # Material cost on actual consumption (per item qty)
                        # If unit_material_cost is per "order", you can instead use a per-item rate map in future.
                        if self.material_cost_mode == "per_consumption":
                            self.cost_material += self.unit_material_cost * required_qty
                        pulled = True
                        break
                if not pulled:
                    # rollback previously pulled items
                    for pb, it, qty in pulled_items:
                        pb.push_item(it, qty)
                    self.starvation_counts[stage.stage_id] += 1
                    self.log.append(f"{self._fmt_t()} '{stage.name}' waiting: insufficient '{item_id}' (need {required_qty}) for model '{model_id}'.")
                    # Preserve model_id in retry
                    self._push_event(self.t + 0.5, "try_start", {
                        "stage_id": stage.stage_id, 
                        "is_rework": is_rework,
                        "model_id": model_id
                    })
                    return

        # Engage team (utilization starts)
        team = self.teams.get(stage.team_id)
        if team:
            team.start_busy(self.t)

        stage.busy = True

        # Calculate processing time:
        # Step 1: Get base time (user-defined, independent of material count)
        base_time = stage.base_process_time_min
        
        # Step 2: Apply deterministic worker efficiency (square root function)
        effective_time = self._apply_worker_efficiency(base_time, stage.workers)
        
        # Step 3: Sample from distribution (this adds variability)
        ptime = sample_time(stage.time_distribution, effective_time)

        # Assign a job_id to this completion (needed if we split into multiple deliveries)
        self._job_seq += 1
        job_id = self._job_seq
        
        # Store model_id for this job
        if model_id:
            self._job_model_map[job_id] = model_id
        
        # Store model-specific outputs and materials for this job
        stage._current_job_outputs = output_buffers
        stage._current_job_model = model_id
        
        # If per-output transport is configured, "complete" means processing done (no transport yet).
        # Otherwise, keep old behavior (processing + transport in one finish_t).
        if getattr(stage, "transport_time_to_outputs_min", None):
            # transport handled by separate deliver events
            finish_t = self.t + ptime
        else:
            finish_t = self.t + ptime + float(stage.transport_time_min or 0.0)

        self._push_event(finish_t, "complete", {
            "stage_id": stage.stage_id, 
            "job_id": job_id,
            "model_id": model_id
        })

        # Remember pulled items for BOM blocking retries
        if bom_mode:
            stage._pulled_items = pulled_items

    def _on_complete(self, ev: Event):
        """Complete processing at a stage, handle defects/rework, then push outputs."""
        stage = self.stages.get(ev.payload.get("stage_id"))
        if stage is None:
            return

        # Release team (utilization ends)
        team = self.teams.get(stage.team_id)
        job_id = ev.payload.get("job_id")
        model_id = ev.payload.get("model_id") or self._job_model_map.get(job_id)

        if team:
            team.stop_busy(self.t)

        # Get model-specific outputs (use stored outputs from try_start if available)
        output_buffers = getattr(stage, "_current_job_outputs", None) or stage.output_buffers.copy()

        # Defect handling (rework or scrap)
        proceed_to_output = True
        if stage.defect_rate and random.random() < stage.defect_rate:
            # Count defects per stage
            self.stage_defect_counts[stage.stage_id] = self.stage_defect_counts.get(stage.stage_id, 0) + 1
            proceed_to_output = False

            if stage.rework_stage_id and stage.rework_stage_id in self.stages:
                # Mark as rework so it bypasses release-stage token gating, preserve model_id
                self._push_event(self.t, "try_start", {
                    "stage_id": stage.rework_stage_id, 
                    "is_rework": True,
                    "model_id": model_id
                })
                self.log.append(
                    f"{self._fmt_t()} '{stage.name}' defect → rework at '{stage.rework_stage_id}' (model '{model_id}')."
                )
            else:
                # Scrap: item leaves the system -> reduce WIP and (FIFO) drop its release time
                self._accumulate_wip(self.t)
                self.current_wip = max(0, self.current_wip - 1)
                self.log.append(f"{self._fmt_t()} '{stage.name}' defect → scrapped item (model '{model_id}').")
                if self._release_times:
                    self._release_times.popleft()

        chosen_out = None
        if proceed_to_output:
            # --- NEW: per-output transport (e.g. S2 -> C1/C2/C3) ---
            if getattr(stage, "transport_time_to_outputs_min", None):
                outputs = list(output_buffers.items())
                self._pending_deliveries[job_id] = {
                    "stage_id": stage.stage_id, 
                    "remaining": len(outputs),
                    "model_id": model_id
                }

                for out_buffer_id, materials in outputs:
                    delay = float(
                        stage.transport_time_to_outputs_min.get(out_buffer_id, stage.transport_time_min) or 0.0)
                    self._push_event(self.t + delay, "deliver", {
                        "stage_id": stage.stage_id,
                        "job_id": job_id,
                        "out_buffer_id": out_buffer_id,
                        "materials": materials,
                        "model_id": model_id
                    })

                # IMPORTANT:
                # Do NOT free the stage here. We will free it after all deliveries succeed.
                return

            all_push_success = True
            pushed_buffers: List[str] = []
            outputs_logged: List[tuple] = []
            for out_buffer_id, materials in output_buffers.items():
                ob = self.buffers.get(out_buffer_id)
                if not ob:
                    self.log.append(f"{self._fmt_t()} '{stage.name}' error: missing output buffer '{out_buffer_id}'.")
                    all_push_success = False
                    continue
                for item_id, qty in materials.items():
                    if not ob.push_item(item_id, qty):
                        # blocking
                        self.blocking_counts[stage.stage_id] += 1
                        self.log.append(f"{self._fmt_t()} '{stage.name}' output blocked: '{out_buffer_id}' full.")
                        self._push_event(self.t + 0.5, "complete", {
                            "stage_id": stage.stage_id,
                            "job_id": job_id,
                            "model_id": model_id
                        })
                        return
                    pushed_buffers.append(out_buffer_id)
                    outputs_logged.append((out_buffer_id, item_id, qty))
                    self.log.append(f"{self._fmt_t()} '{stage.name}' pushed {qty}x '{item_id}' → '{out_buffer_id}' (model '{model_id}').")
                    if str(out_buffer_id) in [str(x) for x in self.finished_buffers]:
                        self._on_unit_finished(qty, model_id=model_id)
                        self.log.append(
                            f"{self._fmt_t()} Product (model '{model_id}') finished into buffer '{out_buffer_id}'. Finished={self.finished}")
            if all_push_success:
                self.stage_completed_counts[stage.stage_id] += 1
                chosen_out = pushed_buffers[0] if pushed_buffers else None

        # Free the stage and immediately attempt next start at this stage
        stage.busy = False
        # Clear stored job-specific data
        if hasattr(stage, "_current_job_outputs"):
            delattr(stage, "_current_job_outputs")
        if hasattr(stage, "_current_job_model"):
            delattr(stage, "_current_job_model")
        self._schedule_try_start(stage.stage_id, delay=0.0, is_rework=False)
        # Wake consumers of pushed buffers - pass model_id if available
        for s in self.stages.values():
            for b in output_buffers.keys():
                if b in s.input_buffers and not s.busy:
                    # Try to pass model_id to downstream stages
                    self._push_event(self.t, "try_start", {
                        "stage_id": s.stage_id, 
                        "is_rework": False,
                        "model_id": model_id
                    })

        # Assembly trace logging (consumed → produced)
        if self.trace_assembly and proceed_to_output:
            consumed_summary = {}
            for _, item_id, qty in getattr(stage, "_pulled_items", []):
                consumed_summary[item_id] = consumed_summary.get(item_id, 0) + qty
            produced_summary = {}
            for ob_id, item_id, qty in outputs_logged if 'outputs_logged' in locals() else []:
                produced_summary.setdefault(ob_id, {})
                produced_summary[ob_id][item_id] = produced_summary[ob_id].get(item_id, 0) + qty
            self.assembly_traces.append({
                "t": self.t,
                "stage_id": stage.stage_id,
                "stage_name": stage.name,
                "consumed": consumed_summary,
                "produced": produced_summary,
            })

#事件处理函数
    def _on_deliver(self, ev: Event):
        """Deliver transported outputs into the destination buffer after a delay."""
        stage_id = ev.payload.get("stage_id")
        job_id = ev.payload.get("job_id")
        out_buffer_id = ev.payload.get("out_buffer_id")
        materials = ev.payload.get("materials") or {}
        model_id = ev.payload.get("model_id") or self._job_model_map.get(job_id)

        stage = self.stages.get(stage_id)
        ob = self.buffers.get(out_buffer_id)
        if stage is None or ob is None:
            self.log.append(f"{self._fmt_t()} deliver error: missing stage/buffer ({stage_id} -> {out_buffer_id}).")
            return

        # --- Atomic capacity check (avoid partial push then double-push on retry) ---
        total_qty = sum(int(q) for q in materials.values())
        if ob.capacity is not None:
            if (ob.total_items() + total_qty) > int(ob.capacity):
                self.blocking_counts[stage_id] += 1
                self.log.append(f"{self._fmt_t()} deliver blocked: '{out_buffer_id}' full (retry).")
                self._push_event(self.t + 0.5, "deliver", ev.payload)
                return

        # Push all items
        for item_id, qty in materials.items():
            ok = ob.push_item(item_id, int(qty))
            if not ok:
                # This should be rare due to atomic check, but keep safe retry
                self.blocking_counts[stage_id] += 1
                self.log.append(f"{self._fmt_t()} deliver blocked (unexpected): '{out_buffer_id}' full (retry).")
                self._push_event(self.t + 0.5, "deliver", ev.payload)
                return
            self.log.append(f"{self._fmt_t()} delivered {qty}x '{item_id}' → '{out_buffer_id}'.")

            if str(out_buffer_id) in [str(x) for x in self.finished_buffers]:
                self._on_unit_finished(int(qty), model_id=model_id)
                self.log.append(
                    f"{self._fmt_t()} Product (model '{model_id}') finished into buffer '{out_buffer_id}'. Finished={self.finished}")


        # Wake consumers of this buffer immediately (do not wait for the last delivery)
        # Pass model_id to downstream stages
        for s in self.stages.values():
            if out_buffer_id in s.input_buffers:
                self._push_event(self.t, "try_start", {
                    "stage_id": s.stage_id, 
                    "is_rework": False,
                    "model_id": model_id
                })

        # Countdown remaining deliveries for this job
        rec = self._pending_deliveries.get(job_id)
        if rec:
            rec["remaining"] -= 1
            if rec["remaining"] <= 0:
                self._pending_deliveries.pop(job_id, None)
                self.stage_completed_counts[stage_id] += 1
                stage.busy = False

                # Re-trigger after the stage is actually freed
                self._schedule_try_start(stage.stage_id, delay=0.0, is_rework=False)

                # Wake consumers of THIS buffer - pass model_id
                for s in self.stages.values():
                    if out_buffer_id in s.input_buffers:
                        self._push_event(self.t, "try_start", {
                            "stage_id": s.stage_id, 
                            "is_rework": False,
                            "model_id": model_id
                        })

        # --------------------------------------------------------------------------
    # Internal time advance
    # --------------------------------------------------------------------------

    def _on_time_advance(self, new_t: float):
        new_t = max(new_t, self.t)
        if new_t == self.t:
            return
        # Sample timeline at fixed intervals up to new_t
        if self._sample_dt and self._sample_dt > 0:
            while self._next_sample_t <= new_t:
                self._sample_snapshot(self._next_sample_t)
                self._next_sample_t += self._sample_dt
        self._accumulate_buffers(new_t)
        self._accumulate_wip(new_t)
        self.t = new_t

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _fmt_t(self) -> str:
        return f"[t={self.t:.2f}{self.time_unit}]"

    def _sample_snapshot(self, at_t: float):
        # Compute throughput per minute since last sample (dt is already in minutes)
        dt = max(1e-9, at_t - self._last_sample_time)
        finished_delta = self.finished - self._last_sample_finished
        throughput_per_min = finished_delta / dt  # dt is in minutes, so this gives units per minute
        snap: Dict[str, Any] = {
            "t": round(at_t, 6),
            "wip": int(self.current_wip),
            "finished": int(self.finished),
            "throughput_per_min": float(throughput_per_min),
        }
        # Buffer levels snapshot
        for b_id, buf in self.buffers.items():
            snap[str(b_id)] = int(buf.total_items())
        self.timeline.append(snap)
        self._last_sample_finished = self.finished
        self._last_sample_time = at_t

    def _accumulate_buffers(self, new_t: float):
        """Accumulate buffer inventory*time area for holding cost and average levels."""
        dt = max(0.0, new_t - self.last_buffer_time)
        if dt <= 0:
            return
        for b_id, buf in self.buffers.items():
            self.buffer_time_area[b_id] += buf.total_items() * dt
        self.last_buffer_time = new_t


# ==============================================================================
# Built-in environment configuration (EDIT HERE)
# ==============================================================================

CONFIG: Dict[str, Any] = {
    # ----------------------------------------------------------------------------
    # Buffers (inventories). Itemized stocks.
    # ----------------------------------------------------------------------------
    "buffers": [
        {"buffer_id": "B",  "name": "Sorted Parts Buffer", "capacity": 9999, "initial_stock": {}},
        {
            "buffer_id": "A",
            "name": "Warehouse A (Bricks)",
            "capacity": 99999,
            "initial_stock": {
                "a01": 1500, "a02": 1500, "a03": 1500, "a04": 1500, "a05": 1500, "a06": 1500, "a07": 1500,
                "b01": 1500, "b02": 1500, "b03": 1500, "b04": 1500, "b05": 1500, "b06": 1500, "b07": 1500,
                "b08": 1500, "b09": 1500, "b10": 1500, "b11": 1500, "b12": 1500, "b13": 1500, "b14": 1500,
                "b15": 1500, "b16": 1500, "b17": 1500, "b18": 1500, "b19": 1500,
                "c01": 1500, "c02": 1500, "c03": 1500, "c04": 1500, "c05": 1500, "c06": 1500, "c07": 1500, "c08": 1500,
                "x01": 1500, "x02": 1500, "x03": 1500, "x04": 1500
            }
        },
        {"buffer_id": "C1", "name": "C1 (Set for Axis Assembly)", "capacity": 9999, "initial_stock": {"buna01": 0, "buna02": 0, "buna03": 0, "buna04": 0}},
        {"buffer_id": "C2", "name": "C2 (Set for Chassis Assembly)", "capacity": 9999, "initial_stock": {"bunc01": 0, "bunc02": 0, "bunc03": 0, "bunc04": 0}},
        {"buffer_id": "C3", "name": "C3 (Final Assembly Only Parts)", "capacity": 9999, "initial_stock": {"bunf01": 0, "bunf02": 0, "bunf03": 0, "bunf04": 0}},
        {"buffer_id": "D1", "name": "D1 (Axis Subassembly)", "capacity": 9999, "initial_stock": {"saa01": 0, "saa02": 0}},
        {"buffer_id": "D2", "name": "D2 (Chassis Subassembly)", "capacity": 9999, "initial_stock": {"sac01": 0, "sac02": 0}},
        {"buffer_id": "E",  "name": "E (Finished Gliders)", "capacity": 9999, "initial_stock": {"fg01": 0, "fg02": 0, "fg03": 0, "fg04": 0}},
    ],

    # ----------------------------------------------------------------------------
    # Teams (workers). One stage uses one team at a time.
    # ----------------------------------------------------------------------------
    "teams": [
        {"team_id": "T1", "name": "Type Sorting Team",  "size": 2, "shift_id": "day"},
        {"team_id": "T2", "name": "Set Sorting Team",   "size": 2, "shift_id": "day"},
        {"team_id": "T3", "name": "Axis Team",          "size": 2, "shift_id": "day"},
        {"team_id": "T4", "name": "Chassis Team",       "size": 2, "shift_id": "day"},
        {"team_id": "T5", "name": "Final Assembly Team","size": 3, "shift_id": "day"},
    ],

    # ----------------------------------------------------------------------------
    # Stages (BOM-based deterministic process)
    # ----------------------------------------------------------------------------
    "stages": [
        {
            "stage_id": "S1",
            "name": "Type Sorting (Classification)",
            "team_id": "T1",
            "input_buffers": ["A"],
            # Model-specific input materials (optional - overrides required_materials when model_id matches)
            "required_materials_by_model": {
                "m01": {"a01": 1, "a03": 2, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b02": 2, "b04": 1, "b05": 2, "b07": 1,
                "b09": 1, "b10": 2, "b11": 1, "b13": 1, "b14": 2,
                "b16": 1, "b18": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c05": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x03": 2},  # Sly_Slider specific materials

                "m02": {"a02": 2, "a04": 1, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b02": 2, "b04": 1, "b05": 2, "b07": 1,
                "b09": 1, "b10": 2, "b11": 1, "b13": 1, "b14": 2,
                "b16": 1, "b18": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c05": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x03": 1, "x04": 1},  # Gliderlinski specific materials

                "m03": {"a02": 2, "a04": 1, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b03": 2, "b04": 1, "b06": 2, "b08": 1, 
                "b09": 1, "b10": 2, "b12": 1, "b13": 1, "b15": 2, 
                "b17": 1, "b19": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c06": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x04": 2},  # Icky_Ice_Glider specific materials

                "m04": {"a01": 1, "a03": 2, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b03": 2, "b04": 1, "b06": 2, "b08": 1, 
                "b09": 1, "b10": 2, "b12": 1, "b13": 1, "b15": 2, 
                "b17": 1, "b19": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c06": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x03": 1, "x04": 1,
                }   # Icomat_2000X specific materials
            },
            
            # Model-specific outputs (optional - different models may produce different sorted parts)
            "output_buffers_by_model": {
                # Structure: model_id -> {buffer_id: {item_id: qty}}
                # All models output to buffer "B" but with different materials
                "m01": {"B": {"a01": 1, "a03": 2, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b02": 2, "b04": 1, "b05": 2, "b07": 1,
                "b09": 1, "b10": 2, "b11": 1, "b13": 1, "b14": 2,
                "b16": 1, "b18": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c05": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x03": 2}},  # Sly_Slider - output to buffer "B"

                "m02": {"B": {"a02": 2, "a04": 1, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b02": 2, "b04": 1, "b05": 2, "b07": 1,
                "b09": 1, "b10": 2, "b11": 1, "b13": 1, "b14": 2,
                "b16": 1, "b18": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c05": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x03": 1, "x04": 1}},  # Gliderlinski - output to buffer "B"

                "m03": {"B": {"a02": 2, "a04": 1, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b03": 2, "b04": 1, "b06": 2, "b08": 1, 
                "b09": 1, "b10": 2, "b12": 1, "b13": 1, "b15": 2, 
                "b17": 1, "b19": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c06": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x04": 2}},  # Icky_Ice_Glider specific materials

                "m04": {"B": {"a01": 1, "a03": 2, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b03": 2, "b04": 1, "b06": 2, "b08": 1, 
                "b09": 1, "b10": 2, "b12": 1, "b13": 1, "b15": 2, 
                "b17": 1, "b19": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c06": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x03": 1, "x04": 1}}  # Icomat_2000X - output to buffer "B"
            },
            "base_process_time_min": 30.0,  # 30 minutes base time
            "time_distribution": {"type": "triangular", "p1": 20.0, "p2": 30.0, "p3": 45.0},  # minutes
            "transport_time_min": 3.0,  # 3 minutes transport
            "defect_rate": 0.0,
            "rework_stage_id": "S1",
            "workers": 2
        },
        {
            "stage_id": "S2",
            "name": "Set Sorting (Kit Build)",
            "team_id": "T2",
            "input_buffers": ["B"],
            # Model-specific input materials
            "required_materials_by_model": {
                "m01": {"a01": 1, "a03": 2, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b02": 2, "b04": 1, "b05": 2, "b07": 1,
                "b09": 1, "b10": 2, "b11": 1, "b13": 1, "b14": 2,
                "b16": 1, "b18": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c05": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x03": 2},  # Sly_Slider specific materials

                "m02": {"a02": 2, "a04": 1, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b02": 2, "b04": 1, "b05": 2, "b07": 1,
                "b09": 1, "b10": 2, "b11": 1, "b13": 1, "b14": 2,
                "b16": 1, "b18": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c05": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x03": 1, "x04": 1},  # Gliderlinski specific materials

                "m03": {"a02": 2, "a04": 1, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b03": 2, "b04": 1, "b06": 2, "b08": 1, 
                "b09": 1, "b10": 2, "b12": 1, "b13": 1, "b15": 2, 
                "b17": 1, "b19": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c06": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x04": 2},  # Icky_Ice_Glider specific materials

                "m04": {"a01": 1, "a03": 2, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b03": 2, "b04": 1, "b06": 2, "b08": 1, 
                "b09": 1, "b10": 2, "b12": 1, "b13": 1, "b15": 2, 
                "b17": 1, "b19": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c06": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x03": 1, "x04": 1,
                }   # Icomat_2000X specific materials
            },
            # Default outputs
            "output_buffers": {
                "C1": {"buna01": 1},   # Axis set
                "C2": {"bunc01": 1},   # Chassis set
                "C3": {"bunf01": 1}    # Final set
            },
            # Model-specific outputs (optional - different models may produce different kits)
            "output_buffers_by_model": {
                "m01": {"C1": {"buna01": 1}, "C2": {"bunc01": 1}, "C3": {"bunf01": 1}},
                "m02": {"C1": {"buna02": 1}, "C2": {"bunc02": 1}, "C3": {"bunf02": 1}},
                "m03": {"C1": {"buna03": 1}, "C2": {"bunc03": 1}, "C3": {"bunf03": 1}},
                "m04": {"C1": {"buna04": 1}, "C2": {"bunc04": 1}, "C3": {"bunf04": 1}}
            },
            "base_process_time_min": 30.0,  # 30 minutes base time
            "time_distribution": {"type": "triangular", "p1": 20.0, "p2": 30.0, "p3": 45.0},  # minutes
            "transport_time_min": 3.0,  # 3 minutes transport
            "defect_rate": 0.0,
            "rework_stage_id": "S2",
            "workers": 2
        },
        {
            "stage_id": "S3",
            "name": "Axis Subassembly",
            "team_id": "T3",
            "input_buffers": ["C1"],
            # Model-specific input materials
            "required_materials_by_model": {
                "m01": {"buna01": 1},  # Sly_Slider
                "m02": {"buna02": 1},  # Gliderlinski
                "m03": {"buna03": 1},  # Icky_Ice_Glider
                "m04": {"buna04": 1}   # Icomat_2000X
            },
            # Default outputs
            "output_buffers": {"D1": {"saa01": 1}},
            # Model-specific outputs (optional - different models may produce different subassemblies)
            "output_buffers_by_model": {
                "m01": {"D1": {"saa01": 1}},  # Sly_Slider
                "m02": {"D1": {"saa02": 1}},  # Gliderlinski
                "m03": {"D1": {"saa02": 1}},  # Icky_Ice_Glider
                "m04": {"D1": {"saa01": 1}}   # Icomat_2000X
            },
            "base_process_time_min": 60.0,  # 60 minutes (1 hour) base time
            "time_distribution": {"type": "triangular", "p1": 45.0, "p2": 60.0, "p3": 80.0},  # minutes
            "transport_time_min": 4.0,  # 4 minutes transport
            "defect_rate": 0.0,
            "rework_stage_id": "S3",
            "workers": 2
        },
        {
            "stage_id": "S4",
            "name": "Chassis Subassembly",
            "team_id": "T4",
            "input_buffers": ["C2"],
            # Model-specific input materials
            "required_materials_by_model": {
                "m01": {"bunc01": 1},  # Sly_Slider
                "m02": {"bunc02": 1},  # Gliderlinski
                "m03": {"bunc03": 1},  # Icky_Ice_Glider
                "m04": {"bunc04": 1}   # Icomat_2000X
            },
            # Default outputs
            "output_buffers": {"D2": {"sac01": 1}},
            # Model-specific outputs (optional - different models may produce different subassemblies)
            "output_buffers_by_model": {
                "m01": {"D2": {"sac01": 1}},  # Sly_Slider
                "m02": {"D2": {"sac01": 1}},  # Gliderlinski
                "m03": {"D2": {"sac02": 1}},  # Icky_Ice_Glider
                "m04": {"D2": {"sac02": 1}}   # Icomat_2000X
            },
            "base_process_time_min": 90.0,  # 90 minutes (1.5 hours) base time
            "time_distribution": {"type": "triangular", "p1": 70.0, "p2": 90.0, "p3": 120.0},  # minutes
            "transport_time_min": 4.0,  # 4 minutes transport
            "defect_rate": 0.0,
            "rework_stage_id": "S4",
            "workers": 2
        },
        {
            "stage_id": "S5",
            "name": "Final Assembly",
            "team_id": "T5",
            "input_buffers": ["C3", "D1", "D2"],
            # Model-specific materials
            "required_materials_by_model": {
                "m01": {"bunf01": 1, "saa01": 1, "sac01": 1},  # Sly_Slider
                "m02": {"bunf02": 1, "saa02": 1, "sac01": 1},  # Gliderlinski
                "m03": {"bunf03": 1, "saa02": 1, "sac02": 1},  # Icky_Ice_Glider
                "m04": {"bunf04": 1, "saa01": 1, "sac02": 1}   # Icomat_2000X
            },
            # Default outputs
            "output_buffers": {"E": {"fg01": 1}},
            # Model-specific outputs (optional - different models produce different finished goods)
            "output_buffers_by_model": {
                "m01": {"E": {"fg01": 1}},  # Sly_Slider
                "m02": {"E": {"fg02": 1}},  # Gliderlinski
                "m03": {"E": {"fg03": 1}},  # Icky_Ice_Glider
                "m04": {"E": {"fg04": 1}}    # Icomat_2000X
            },
            "base_process_time_min": 120.0,  # 120 minutes (2 hours) base time
            "time_distribution": {"type": "triangular", "p1": 90.0, "p2": 120.0, "p3": 150.0},  # minutes
            "transport_time_min": 5.0,  # 5 minutes transport
            "defect_rate": 0.0,
            "rework_stage_id": "S5",
            "workers": 2
        }
    ],

    # ----------------------------------------------------------------------------
    # Shift schedule (optional). Times are minutes in a 24h day.
    # For 3-week continuous simulation, production runs 24/7 (disabled shifts).
    # To enable 8-hour daily shifts, uncomment the line below.
    # ----------------------------------------------------------------------------
    "shift_schedule": [],  # Disabled for continuous 3-week production
    # "shift_schedule": [{"shift_id": "day", "start_minute": 8 * 60, "end_minute": 16 * 60}],  # 8-hour daily shifts

    # ----------------------------------------------------------------------------
    # Model definitions (optional - for multi-model support)
    # Each model can have different material requirements at different stages
    # ----------------------------------------------------------------------------
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
    },

    # ----------------------------------------------------------------------------
    # Global parameters (optional, free-form)
    # ----------------------------------------------------------------------------
    "parameters": {
        "default_model_id": "m01",  # Default model if not specified
        "target_takt_min": 10.0,  # Target takt time in minutes
        "timeline_sample_dt_min": 5.0,  # Timeline sampling interval in minutes
        "finished_buffer_ids": ["E"],
        # Model demand probability distribution (for forecast allocation)
        "model_demand_probabilities": {
            "m01": 0.60,  # Sly_Slider 60%
            "m03": 0.20,  # Icky_Ice_Glider 20%
            "m02": 0.10,  # Gliderlinski 10%
            "m04": 0.10   # Icomat_2000X 10%
        },
        # Cost and revenue settings (all optional; units in currency/second where applicable)
        "cost": {
            "unit_price": 10000.0,           # revenue per finished glider
            "unit_material_cost": 4400.0,    # material cost per released order (can be adjusted to per-BOM)
            "labor_costs_per_team_sec": {    # cost rate per team per second (multiplied by team size)
                "T1": 0.010,
                "T2": 0.010,
                "T3": 0.010,
                "T4": 0.010,
                "T5": 0.012
            },
            "holding_costs_per_buffer_sec": { # cost rate per unit inventory per second
                "A": 0.0005, "B": 0.0005,
                "C1": 0.001,
                "C2": 0.001,
                "C3": 0.001,
                "D1": 0.001,
                "D2": 0.001,
                "E": 0.0005
            },
            "margin_per_unit": 5600.0,       # gross margin per finished glider (before costs)
            "demand_qty": None               # optional cap on sellable units (None = unlimited)
        },
        "conwip_wip_cap": 12,  # 全局 WIP 上限（先用 8~15 试，建议从 12 起步）
        "auto_release_conwip": True,  # 成品出系统后自动补放行
        "kanban_caps": {  # 关键缓冲 Kanban 上限（建议先只控合流相关）
            "C3": 4,
            "D1": 4,
            "D2": 4
        },
        # Push demand planning (finite push)
        "push_demand_enabled": True,
        "push_auto_release": False,
        "push_demand_horizon_weeks": 3,
        "push_weekly_demand_mean": 25,
        "push_forecast_noise_pct": 0.1,
        "push_realization_noise_pct": 0.05,
        "push_procurement_waste_rate": 0.05,
        "supplier_stage_ids": [],  # S1 should process materials normally (pull from A, output to B), not act as supplier
        "material_cost_mode": "procure_forecast",
    },

}
# ==============================================================================
# Minimal example (you can delete this block in production)
# ==============================================================================
if __name__ == "__main__":
    # Build env with deterministic seed for reproducibility
    env = LegoLeanEnv(CONFIG, time_unit="sec", seed=42)

    # Release some orders into the system (source stage = S1)
    # Example: Release orders with model types
    env.enqueue_orders(qty=50, model_id="m01")  # Release 50 orders of m01 (Sly_Slider)
    # You can also mix models: env.enqueue_orders(qty=10, model_id="m02")  # Then 10 orders of m02

    # Kick off "try_start" for all stages once (source stages already enqueued above)
    for s in env.stages.values():
        env._push_event(env.t, "try_start", {"stage_id": s.stage_id})

    # Simulate for 3 weeks (3 * 7 * 24 * 60 = 30240 minutes)
    env.run_for(30240.0)

    # KPIs
    print("--- KPIs ---")
    for k, v in env.get_kpis().items():
        print(f"{k}: {v}")

    # Trace (last few lines)
    print("\n--- Trace (last 15) ---")
    for line in env.log[-15:]:
        print(line)
