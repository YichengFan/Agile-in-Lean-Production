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
# - Keep units consistent. This file uses **seconds** for simulated time.
# --------------------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import heapq
import random
import math
import time
from collections import deque  # Used to track order release times for lead-time calculation



# ==============================================================================
# Utility helpers
# ==============================================================================

def wall_ms() -> int:
    """Wall-clock milliseconds (only for profiling/logging, not simulation time)."""
    return int(time.time() * 1000)


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
    - workers: number of workers at this stage. Processing time scales with total required qty.
    - Supports model-specific materials: required_materials_by_model (model_id -> {item_id: qty})
    """
    stage_id: Any
    name: str
    team_id: Any
    input_buffers: List[Any] = field(default_factory=list)  # e.g., ['D1', 'D2', 'C3'] for Final Assembly
    required_materials: Dict[str, int] = field(default_factory=dict)  # BOM-style requirements (item_id -> qty) - legacy/default
    required_materials_by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)  # model_id -> {item_id: qty}
    output_buffers: Dict[str, Dict[str, int]] = field(default_factory=dict)  # deterministic outputs with items
    output_buffers_by_model: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)  # model_id -> {buffer_id: {item_id: qty}}

    base_process_time_sec: float = 1.0
    time_distribution: Dict[str, Any] = field(default_factory=dict)
    transport_time_sec: float = 0.0
    #dic to save the time from s2 to c1,c2,c3
    transport_time_to_outputs_sec: dict = field(default_factory=dict)
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

    def __init__(self, config: Dict[str, Any], time_unit: str = "sec", seed: Optional[int] = None):
        self.cfg = config
        self.time_unit = time_unit
        # 2026-01-01 把tracelog初始化弄上来一点，这样在原本位置上面的日志就可以正常输出了
        # Trace log
        self.log: List[str] = []
        if seed is not None:
            random.seed(seed)

        # Build buffers
        self.buffers: Dict[Any, Buffer] = {}
        for b in self.cfg.get("buffers", []):
            b_id = b.get("buffer_id") or b.get("name")
            self.buffers[b_id] = Buffer(
                buffer_id=b_id,
                name=b.get("name") or str(b_id),
                capacity=b.get("capacity"),
                initial_stock=b.get("initial_stock") or 0
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
            self.stages[s_id] = Stage(
                stage_id=s_id,
                name=s.get("name") or str(s_id),
                team_id=s.get("team_id"),
                input_buffers=in_bufs,
                required_materials=s.get("required_materials") or {},
                required_materials_by_model=s.get("required_materials_by_model") or {},
                output_buffers=s.get("output_buffers") or {},
                output_buffers_by_model=s.get("output_buffers_by_model") or {},
                base_process_time_sec=float(s.get("base_process_time_sec") or 1.0),
                time_distribution=s.get("time_distribution") or {},
                transport_time_sec=float(s.get("transport_time_sec") or 0.0),
                #读取 config 时把这个字段读进 Stage
                # 2026-01-01 你自己看看这条还要不要把，你如果要的话，你config里也没那个玩意儿，我感觉你这个代码是不是直接照抄GPT啊
                # 他写这个or 你就跟了虽然面板上能看到但实际上这只不过是把原来 transport_tim_sec复制进去了，请你检查这样的操作是否
                # 真的可以激活765的判定，或者你看到后面我写的注释你自己再思考一下是不是应该直接合并所有运输都在deliver里面而不只是S2
                transport_time_to_outputs_sec=(s.get("transport_time_to_outputs_sec") or {}).copy(),
                defect_rate=float(s.get("defect_rate") or 0.0),
                rework_stage_id=s.get("rework_stage_id"),
                workers=int(s.get("workers") or 1)
            )

        # Shifts (optional)
        self.shifts = self.cfg.get("shift_schedule", [])

        # Global parameters and random disruption controls
        self.parameters = self.cfg.get("parameters", {})
        self.random_events = self.cfg.get("random_events", {})
        self.finished_buffers = list(self.parameters.get("finished_buffer_ids", ["E"]))
        self.trace_assembly: bool = bool(self.parameters.get("trace_assembly", False))
        # Cost & revenue parameters (optional)
        cost_cfg = self.parameters.get("cost", {})
        self.unit_price = float(cost_cfg.get("unit_price", 0.0))
        self.unit_material_cost = float(cost_cfg.get("unit_material_cost", 0.0))
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
        #self.release_stage_ids = set(self.parameters.get("release_stage_ids", []))

        # 2026-01-01 手动设置起始站点为S1因为发现不知道为什么parameter读不出来，反正没事log反应是对的就好
        self.release_stage_ids = {"S1"}
        self.log.append(f"Release stages detected: {self.release_stage_ids}")


        # Global WIP cap for CONWIP; if None, CONWIP is disabled
        self.conwip_wip_cap = self.parameters.get("conwip_wip_cap", None)

        # If True, each finished unit automatically releases one new order (closed-loop CONWIP)
        self.auto_release_conwip = bool(self.parameters.get("auto_release_conwip", False))

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

        #2026-01-01 tracelog 弄上去了

        # Timeline sampling
        self.timeline: List[Dict[str, Any]] = []
        self._sample_dt: float = float(self.parameters.get("timeline_sample_dt_sec", 5.0) or 5.0)
        self._next_sample_t: float = self._sample_dt
        self._last_sample_finished: int = 0
        self._last_sample_time: float = 0.0
        # Assembly traces (optional)
        self.assembly_traces: List[Dict[str, Any]] = []

        # If current time not in shift, auto-align to next shift start
        if self.shifts and not self._is_in_shift(0.0):
            self.t = self._advance_to_next_shift_start(0.0)
            self.last_wip_time = self.t
            self.last_buffer_time = self.t

        # KPI counters per stage
        self.stage_completed_counts: Dict[Any, int] = {s_id: 0 for s_id in self.stages}
        self.starvation_counts: Dict[Any, int] = {s_id: 0 for s_id in self.stages}
        self.blocking_counts: Dict[Any, int] = {s_id: 0 for s_id in self.stages}
        self.stage_defect_counts: Dict[Any, int] = {s_id: 0 for s_id in self.stages}

    # --------------------------------------------------------------------------
    # Shift logic
    # --------------------------------------------------------------------------

    def _is_in_shift(self, t: float) -> bool:
        """Return True if time t (sec) falls inside any active shift window."""
        if not self.shifts:
            return True
        minute_in_day = (t / 60.0) % 1440.0
        for sh in self.shifts:
            start_min = float(sh.get("start_minute", 0))
            end_min = float(sh.get("end_minute", 1440))
            if start_min <= minute_in_day <= end_min:
                return True
        return False

    def _advance_to_next_shift_start(self, t: float) -> float:
        """Return the next time (sec) when a shift starts after t."""
        if not self.shifts:
            return t
        minute_in_day = (t / 60.0) % 1440.0
        starts = sorted([float(s.get("start_minute", 0)) for s in self.shifts])
        for st in starts:
            if st > minute_in_day:
                delta_min = st - minute_in_day
                return t + delta_min * 60.0
        # Wrap to next day's first shift
        delta_min = (1440.0 - minute_in_day) + starts[0]
        return t + delta_min * 60.0

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


    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def enqueue_orders(self, qty: int = 1, model_id: Optional[str] = None):
        """
        Release 'qty' orders into the system (CONWIP-aware).
        Adds order tokens to release stages (e.g., S1) and triggers try_start.
        
        Args:
            qty: Number of orders to release
            model_id: Model type identifier (e.g., "M1", "M2", "M3", "M4"). 
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
        cap = self.parameters.get("conwip_wip_cap", None)
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

        # Respect shift windows
        if not self._is_in_shift(ev.time):
            shifted = self._advance_to_next_shift_start(ev.time)
            self._push_event(shifted, ev.kind, ev.payload)
            # Continue processing the next event (don't skip it)
            return self.step()

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
        """Run the simulation for dt seconds from current time."""
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

    def _on_unit_finished(self, qty: int = 1):
        """Called when finished units enter a finished buffer (e.g., E).

        Updates:
          - finished count
          - lead times (FIFO match to release times)
          - WIP (decrement on completion)
          - optionally auto-releases CONWIP orders
        """
        qty = int(qty)
        if qty <= 0:
            return

        self.finished += qty

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
        inventory_cost = 0.0
        for b_id, area in self.buffer_time_area.items():
            h = self.holding_costs_per_buffer.get(str(b_id), 0.0)
            inventory_cost += h * area

        # Revenue (all finished are assumed sold unless demand cap is set)
        sales_units = self.finished if self.demand_qty is None else min(self.finished, self.demand_qty)
        revenue_total = self.unit_price * sales_units

        cost_total = self.cost_material + labor_cost + inventory_cost + self.cost_other
        profit = revenue_total - cost_total

        avg_buffer_levels = {
            str(b_id): (area / sim_time) if sim_time > 0 else 0.0
            for b_id, area in self.buffer_time_area.items()
        }
        # Persist latest cost/revenue snapshots for external inspection
        self.cost_labor = labor_cost
        self.cost_inventory = inventory_cost
        self.revenue_total = revenue_total
        return {
            "sim_time_sec": sim_time,
            "throughput_per_sec": throughput,
            "lead_time_avg_sec": lead_time_avg,
            "wip_avg_units": wip_avg,
            "utilization_per_team": utilization,
            "finished_units": self.finished,
            "started_units": self.started,
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
            "avg_buffer_levels": avg_buffer_levels,
        }

    # --------------------------------------------------------------------------
    # Event handlers
    # --------------------------------------------------------------------------

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
            # For non-release stages, try to infer model_id from recent jobs or use default
            # In a real system, you might track model_id per buffer item, but for simplicity
            # we use the default model if not specified
            model_id = self.parameters.get("default_model_id") or "default"
            # Note: In a more sophisticated implementation, you could track which model
            # produced items in each buffer, but for now we use default when not specified

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

        if required_materials:
            for item_id, required_qty in required_materials.items():
                pulled = False
                for b_id in stage.input_buffers:
                    buf = self.buffers.get(b_id)
                    if buf and buf.pull_item(item_id, required_qty):
                        pulled_items.append((buf, item_id, required_qty))
                        # Material cost on actual consumption (per item qty)
                        # If unit_material_cost is per "order", you can instead use a per-item rate map in future.
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

        # Draw processing time from distribution + optional disruption penalty
        total_parts = sum(required_materials.values()) if required_materials else 1
        base_time_per_unit = (stage.base_process_time_sec * max(1, total_parts)) / max(1, stage.workers)
        ptime = sample_time(stage.time_distribution, base_time_per_unit)

        # Random disruption: missing bricks → extra processing time penalty
        missing_prob = float(self.random_events.get("missing_brick_prob", 0.0))
        if random.random() < missing_prob:
            penalty = float(self.random_events.get("missing_brick_penalty_sec", 5.0))
            ptime += penalty
            self.log.append(f"{self._fmt_t()} Disruption at '{stage.name}': missing bricks (+{penalty:.2f}s).")

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
        if getattr(stage, "transport_time_to_outputs_sec", None):
            # transport handled by separate deliver events
            finish_t = self.t + ptime
        else:
            finish_t = self.t + ptime + float(stage.transport_time_sec or 0.0)

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


        # 2026-01-01 请你检查这里真的进的了这个事件吗？如果你要专门为了S2开一个运送事件，是否可以把所有站点的运送/激活都在这个事件中激活？
        # 这样的话可以直接把773 return后面的全注释掉了，因为757行的判定也不那么必要了直接进deliver就可以了，因为我看了你deliver的逻辑
        # deliver逻辑就算一般的站点也可以走这一套逻辑
        chosen_out = None
        if proceed_to_output:
            # --- NEW: per-output transport (e.g. S2 -> C1/C2/C3) ---
            if getattr(stage, "transport_time_to_outputs_sec", None):
                outputs = list(output_buffers.items())
                self._pending_deliveries[job_id] = {
                    "stage_id": stage.stage_id, 
                    "remaining": len(outputs),
                    "model_id": model_id
                }

                for out_buffer_id, materials in outputs:
                    delay = float(
                        stage.transport_time_to_outputs_sec.get(out_buffer_id, stage.transport_time_sec) or 0.0)
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
                        self._push_event(self.t + 0.5, "complete", {"stage_id": stage.stage_id})
                        return
                    pushed_buffers.append(out_buffer_id)
                    outputs_logged.append((out_buffer_id, item_id, qty))
                    self.log.append(f"{self._fmt_t()} '{stage.name}' pushed {qty}x '{item_id}' → '{out_buffer_id}'.")
                    if str(out_buffer_id) in [str(x) for x in self.finished_buffers]:
                        self._on_unit_finished(qty)
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
                self._on_unit_finished(int(qty))
                self.log.append(
                    f"{self._fmt_t()} Product finished into buffer '{out_buffer_id}'. Finished={self.finished}")


        # Wake consumers of this buffer immediately (do not wait for the last delivery)
        # Pass model_id to downstream stages
        for s in self.stages.values():
            if out_buffer_id in s.input_buffers:
                self._push_event(self.t, "try_start", {
                    "stage_id": s.stage_id, 
                    "is_rework": False,
                    "model_id": model_id
                })

        #2026-01-01 更改顺序使得正常触发
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
        # Compute throughput per minute since last sample
        dt = max(1e-9, at_t - self._last_sample_time)
        finished_delta = self.finished - self._last_sample_finished
        throughput_per_min = (finished_delta / dt) * 60.0
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
            "capacity": 9999,
            "initial_stock": {
                "a01": 50, "a02": 50, "a03": 50, "a04": 50, "a05": 50, "a06": 50, "a07": 50,
                "b01": 50, "b02": 50, "b03": 50, "b04": 50, "b05": 50, "b06": 50, "b07": 50,
                "b08": 50, "b09": 50, "b10": 50, "b11": 50, "b12": 50, "b13": 50, "b14": 50,
                "b15": 50, "b16": 50, "b17": 50, "b18": 50, "b19": 50,
                "c01": 50, "c02": 50, "c03": 50, "c04": 50, "c05": 50, "c06": 50, "c07": 50, "c08": 50,
                "x01": 50, "x02": 50, "x03": 50, "x04": 50
            }
        },
        {"buffer_id": "C1", "name": "C1 (Set for Axis Assembly)", "capacity": 9999, "initial_stock": {"bun01": 0}},
        {"buffer_id": "C2", "name": "C2 (Set for Chassis Assembly)", "capacity": 9999, "initial_stock": {"bun02": 0}},
        {"buffer_id": "C3", "name": "C3 (Final Assembly Only Parts)", "capacity": 9999, "initial_stock": {"bun03": 0}},
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
            "required_materials": {
                "a01": 1, "a03": 2, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b02": 2, "b04": 1, "b05": 2, "b07": 1,
                "b09": 1, "b10": 2, "b11": 1, "b13": 1, "b14": 2,
                "b16": 1, "b18": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c05": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x03": 2
            },
            "output_buffers": {
                # After sorting, classified parts go to B (same items)
                "B": {
                    "a01": 1, "a03": 2, "a05": 2, "a06": 1, "a07": 2,
                    "b01": 1, "b02": 2, "b04": 1, "b05": 2, "b07": 1,
                    "b09": 1, "b10": 2, "b11": 1, "b13": 1, "b14": 2,
                    "b16": 1, "b18": 2,
                    "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c05": 4, "c07": 1, "c08": 1,
                    "x01": 3, "x02": 4, "x03": 2
                }
            },
            "base_process_time_sec": 3.0,
            "time_distribution": {"type": "triangular", "p1": 2.0, "p2": 3.0, "p3": 5.0},
            "transport_time_sec": 0.3,
            "defect_rate": 0.01,
            "rework_stage_id": "S1",
            "workers": 2
        },
        {
            "stage_id": "S2",
            "name": "Set Sorting (Kit Build)",
            "team_id": "T2",
            "input_buffers": ["B"],
            # Build three kits from classified parts in B
            # Default materials (used if model_id is "default" or not found in required_materials_by_model)
            "required_materials": {
                "a01": 1, "a03": 2, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b02": 2, "b04": 1, "b05": 2, "b07": 1,
                "b09": 1, "b10": 2, "b11": 1, "b13": 1, "b14": 2,
                "b16": 1, "b18": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c05": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x03": 2
            },
            #Model-specific input materials (optional - overrides required_materials when model_id matches)
            "required_materials_by_model": {
                # Example structure - customize per model as needed:
                "m01": {"a01": 1, "a03": 1},  # Sly_Slider specific materials
                "m02": {"a01": 1, "a03": 1},  # Gliderlinski specific materials
                "m03": {"a01": 1, "a03": 1},  # Icky_Ice_Glider specific materials
                "m04": {"a01": 1, "a03": 1}   # Icomat_2000X specific materials
            },
            # Default outputs
            "output_buffers": {
                "C1": {"bun01": 1},   # Axis set
                "C2": {"bun02": 1},   # Chassis set
                "C3": {"bun03": 1}    # Final set
            },
            # Model-specific outputs (optional - different models may produce different kits)
            "output_buffers_by_model": {
                # Example structure - customize per model as needed:
                "m01": {"C1": {"bun01": 1}, "C2": {"bun02": 1}, "C3": {"bun03": 1}},
                "m02": {"C1": {"bun01": 1}, "C2": {"bun02": 1}, "C3": {"bun03": 1}},
                "m03": {"C1": {"bun01": 1}, "C2": {"bun02": 1}, "C3": {"bun03": 1}},
                "m04": {"C1": {"bun01": 1}, "C2": {"bun02": 1}, "C3": {"bun03": 1}}
            },
            "base_process_time_sec": 3.0,
            "time_distribution": {"type": "triangular", "p1": 2.0, "p2": 3.0, "p3": 5.0},
            "transport_time_sec": 0.3,
            "defect_rate": 0.01,
            "rework_stage_id": "S2",
            "workers": 2
        },
        {
            "stage_id": "S3",
            "name": "Axis Subassembly",
            "team_id": "T3",
            "input_buffers": ["C1"],
            # Default materials
            "required_materials": {"bun01": 1},
            # Model-specific input materials (optional - overrides required_materials when model_id matches)
            "required_materials_by_model": {
                # Example structure - customize per model as needed:
                "m01": {"bun01": 1},  # Sly_Slider
                "m02": {"bun01": 1},  # Gliderlinski
                "m03": {"bun01": 1},  # Icky_Ice_Glider
                "m04": {"bun01": 1}   # Icomat_2000X
            },
            # Default outputs
            "output_buffers": {"D1": {"saa01": 1}},
            # Model-specific outputs (optional - different models may produce different subassemblies)
            "output_buffers_by_model": {
                # Example structure - customize per model as needed:
                "m01": {"D1": {"saa01": 1}},  # Sly_Slider
                "m02": {"D1": {"saa01": 1}},  # Gliderlinski
                "m03": {"D1": {"saa01": 1}},  # Icky_Ice_Glider
                "m04": {"D1": {"saa01": 1}}   # Icomat_2000X
            },
            "base_process_time_sec": 4.0,
            "time_distribution": {"type": "triangular", "p1": 3.0, "p2": 4.0, "p3": 6.0},
            "transport_time_sec": 0.4,
            "defect_rate": 0.02,
            "rework_stage_id": "S3",
            "workers": 2
        },
        {
            "stage_id": "S4",
            "name": "Chassis Subassembly",
            "team_id": "T4",
            "input_buffers": ["C2"],
            # Default materials
            "required_materials": {"bun02": 1},
            # Model-specific input materials (optional - overrides required_materials when model_id matches)
            "required_materials_by_model": {
                # Example structure - customize per model as needed:
                "m01": {"bun02": 1},  # Sly_Slider
                "m02": {"bun02": 1},  # Gliderlinski
                "m03": {"bun02": 1},  # Icky_Ice_Glider 
                "m04": {"bun02": 1}   # Icomat_2000X
            },
            # Default outputs
            "output_buffers": {"D2": {"sac01": 1}},
            # Model-specific outputs (optional - different models may produce different subassemblies)
            "output_buffers_by_model": {
                # Example structure - customize per model as needed:
                "m01": {"D2": {"sac01": 1}},  # Sly_Slider
                "m02": {"D2": {"sac01": 1}},  # Gliderlinski
                "m03": {"D2": {"sac01": 1}},  # Icky_Ice_Glider
                "m04": {"D2": {"sac01": 1}}   # Icomat_2000X
            },
            "base_process_time_sec": 4.0,
            "time_distribution": {"type": "triangular", "p1": 3.0, "p2": 4.0, "p3": 6.0},
            "transport_time_sec": 0.4,
            "defect_rate": 0.02,
            "rework_stage_id": "S4",
            "workers": 2
        },
        {
            "stage_id": "S5",
            "name": "Final Assembly",
            "team_id": "T5",
            "input_buffers": ["C3", "D1", "D2"],
            # Default materials (used if model_id is "default" or not found in required_materials_by_model)
            "required_materials": {"bun03": 1, "saa01": 1, "sac01": 1},
            # Model-specific materials (optional - overrides required_materials when model_id matches)
            # Example: Different models may require different quantities or different parts
            "required_materials_by_model": {
                # Example structure - customize per model as needed:
                "m01": {"bun03": 1, "saa01": 1, "sac01": 1},  # Sly_Slider
                "m02": {"bun03": 2, "saa01": 1, "sac01": 1},  # Gliderlinski: needs extra bun03
                "m03": {"bun03": 1, "saa01": 1, "sac01": 1},  # Icky_Ice_Glider: no sac01
                "m04": {"bun03": 1, "saa01": 1, "sac01": 1}   # Icomat_2000X: double saa01
            },
            # Default outputs
            "output_buffers": {"E": {"fg01": 1}},
            # Model-specific outputs (optional - different models produce different finished goods)
            "output_buffers_by_model": {
                # Example structure - customize per model as needed:
                "m01": {"E": {"fg01": 1}},  # Sly_Slider
                "m02": {"E": {"fg02": 1}},  # Gliderlinski
                "m03": {"E": {"fg03": 1}},  # Icky_Ice_Glider
                "m04": {"E": {"fg04": 1}}    # Icomat_2000X
            },
            "base_process_time_sec": 5.0,
            "time_distribution": {"type": "triangular", "p1": 4.0, "p2": 5.0, "p3": 7.0},
            "transport_time_sec": 0.5,
            "defect_rate": 0.02,
            "rework_stage_id": "S5",
            "workers": 2
        }
    ],

    # ----------------------------------------------------------------------------
    # Shift schedule (optional). Times are minutes in a 24h day. Here: 08:00–16:00
    # ----------------------------------------------------------------------------
    "shift_schedule": [
        {"shift_id": "day", "start_minute": 8 * 60, "end_minute": 16 * 60}
    ],

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
        "target_takt_sec": 10.0,
        "timeline_sample_dt_sec": 5.0,
        "finished_buffer_ids": ["E"],
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
            "demand_qty": None               # optional cap on sellable units (None = unlimited)
        },
        #"release_stage_ids": ["S1"],  # 把 S1 作为入口放行的工序（即使它有 input_buffers 也可以） 2026-01-01 manual setted
        "conwip_wip_cap": 12,  # 全局 WIP 上限（先用 8~15 试，建议从 12 起步）
        "auto_release_conwip": True,  # 成品出系统后自动补放行
        "kanban_caps": {  # 关键缓冲 Kanban 上限（建议先只控合流相关）
            "C3": 4,
            "D1": 4,
            "D2": 4
        }
    },

    # ----------------------------------------------------------------------------
    # Random disruptions (optional)
    # - missing_brick_prob: probability that an operation suffers extra time
    # - missing_brick_penalty_sec: the extra time added when disruption occurs
    # ----------------------------------------------------------------------------
    "random_events": {
        "missing_brick_prob": 0.10,
        "missing_brick_penalty_sec": 2.0
    }
}
# ==============================================================================
# Minimal example (you can delete this block in production)
# ==============================================================================
if __name__ == "__main__":
    # Build env with deterministic seed for reproducibility
    env = LegoLeanEnv(CONFIG, time_unit="sec", seed=42)

    # Release some orders into the system (source stage = S1)
    env.enqueue_orders(qty=50)

    # Kick off "try_start" for all stages once (source stages already enqueued above)
    for s in env.stages.values():
        env._push_event(env.t, "try_start", {"stage_id": s.stage_id})

    # Simulate for 1 hour (3600 seconds)
    env.run_for(3600.0)

    # KPIs
    print("--- KPIs ---")
    for k, v in env.get_kpis().items():
        print(f"{k}: {v}")

    # Trace (last few lines)
    print("\n--- Trace (last 15) ---")
    for line in env.log[-15:]:
        print(line)
