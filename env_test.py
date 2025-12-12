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
    """Storage for multiple item types (tracks quantities per item ID)."""
    buffer_id: Any
    name: str
    capacity: Optional[int] = None
    initial_stock: Optional[Dict[str, int]] = None

    # internal state: item_id → quantity
    items: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize item quantities
        if self.initial_stock:
            # Copy the initial stock dictionary
            self.items = dict(self.initial_stock)
        else:
            self.items = {}

    # Check if buffer can supply specific item(s)
    def can_pull_item(self, item_id: str, qty: int = 1) -> bool:
        return self.items.get(item_id, 0) >= qty

    # Pull specific item(s)
    def pull_item(self, item_id: str, qty: int = 1) -> Optional[Dict[str, int]]:
        if self.can_pull_item(item_id, qty):
            self.items[item_id] -= qty
            if self.items[item_id] <= 0:
                del self.items[item_id]
            return {item_id: qty}
        return None

    # Push specific item(s)
    def push_item(self, item_id: str, qty: int = 1) -> bool:
        if self.capacity is not None and self.total_items() + qty > self.capacity:
            return False
        self.items[item_id] = self.items.get(item_id, 0) + qty
        return True

    # Total stock level (sum of all quantities)
    def total_items(self) -> int:
        return sum(self.items.values())
    # NOTE (for future improvement):
    # When push_item() fails due to capacity limit,
    # downstream blocking must be triggered.
    # This will be handled in _on_complete() via retry logic (blocking event).


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
    Deterministic process node — no randomness.
    Each stage:
      - Pulls exact quantities of required materials (defined in config)
      - Processes them with its team
      - Pushes deterministic outputs to configured buffers
    """

    stage_id: Any
    name: str
    team_id: Any

    # --- Connections ---
    input_buffers: List[str] = field(default_factory=list)
    output_buffers: Dict[str, Dict[str, int]] = field(default_factory=dict)
    """
    Example:
        {"D1": {"axis_subassembly": 1}}
        {"E": {"final_glider": 1}}
    """

    # --- Material Logic ---
    required_materials: Dict[str, int] = field(default_factory=dict)
    """
    Example: {"a01": 4, "a02": 2, "x03": 1}
    Stage cannot start until all required materials are available.
    """

    # --- Timing ---
    base_process_time_sec: float = 1.0
    time_distribution: Dict[str, Any] = field(default_factory=dict)
    transport_time_sec: float = 0.0
    defect_rate: float = 0.0
    rework_stage_id: Optional[str] = None
    workers: int = 1

    # --- Internal State ---
    busy: bool = False
    blocked: bool = False
    output_ready: bool = False
    current_batch: List[tuple] = field(default_factory=list)


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
        if seed is not None:
            random.seed(seed)


        # Trace log
        self.log: List[str] = []

        # ==========================================================
        # BUILD BUFFERS (supports int or dict format)
        # ==========================================================
        self.buffers: Dict[Any, Buffer] = {}

        for b in self.cfg.get("buffers", []):
            b_id = b.get("buffer_id") or b.get("name")
            init_stock = b.get("initial_stock", 0)

            # Case 1: dictionary form → use directly (e.g., {'a01': 3, 'b05': 7})
            if isinstance(init_stock, dict):
                init_materials = {str(k): int(v) for k, v in init_stock.items()}

            # Case 2: integer → generate dummy items (e.g., 30 generic parts)
            elif isinstance(init_stock, int):
                init_materials = {f"{b_id}_item_{i + 1}": 1 for i in range(init_stock)}

            # Default → empty buffer
            else:
                init_materials = {}

            # Create the buffer instance
            self.buffers[b_id] = Buffer(
                buffer_id=b_id,
                name=b.get("name") or str(b_id),
                capacity=b.get("capacity"),
                initial_stock=init_materials
            )

            # Optional log message for debugging
            total_qty = sum(init_materials.values())
            self.log.append(f"{self._fmt_t()} Buffer '{b_id}' initialized with total {total_qty} items.")

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

        # ==========================================================
        # BUILD STAGES
        # ==========================================================
        self.stages: Dict[Any, Stage] = {}

        for s in self.cfg.get("stages", []):
            s_id = s.get("stage_id") or s.get("name")

            # Normalize input buffers: make sure it's always a list
            in_bufs = s.get("input_buffers")
            if isinstance(in_bufs, str) and in_bufs.strip():
                in_bufs = [in_bufs]
            in_bufs = in_bufs or []

            # Deterministic output: dictionary mapping of buffer_id → materials
            out_bufs = s.get("output_buffers")
            if isinstance(out_bufs, dict):
                output_buffers = out_bufs
            elif s.get("output_buffer"):
                output_buffers = {s.get("output_buffer"): {}}
            else:
                output_buffers = {}

            req_mats = s.get("required_materials", {})
            if isinstance(req_mats, list):
                # convert list into dict with qty = 1 (for backward compatibility)
                required_materials = {mid: 1 for mid in req_mats}
            else:
                required_materials = req_mats

            # Create the Stage
            self.stages[s_id] = Stage(
                stage_id=s_id,
                name=s.get("name") or str(s_id),
                team_id=s.get("team_id"),
                input_buffers=in_bufs,
                output_buffers=output_buffers,
                required_materials=required_materials,
                base_process_time_sec=float(s.get("base_process_time_sec") or 1.0),
                time_distribution=s.get("time_distribution") or {},
                transport_time_sec=float(s.get("transport_time_sec") or 0.0),
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
        self.routing_mode: str = str(self.parameters.get("routing_mode", "random")).lower()
        # Routing state for deterministic mode (served counts per stage/output)
        self._served_counts: Dict[Any, Dict[str, int]] = {}
        self._rr_index: Dict[Any, int] = {}

        # Event queue and time
        self._evt_seq = 0
        self._queue: List[Event] = []
        self.t: float = 0.0

        # KPIs
        self.finished: int = 0
        self.started: int = 0
        self.lead_times: List[float] = []
        self.wip_time_area: float = 0.0
        self.last_wip_time: float = 0.0
        self.current_wip: int = 0

        # Trace log
        self.log: List[str] = []

        # Timeline sampling
        self.timeline: List[Dict[str, Any]] = []
        self._sample_dt: float = float(self.parameters.get("timeline_sample_dt_sec", 5.0) or 5.0)
        self._next_sample_t: float = self._sample_dt
        self._last_sample_finished: int = 0
        self._last_sample_time: float = 0.0

        # If current time not in shift, auto-align to next shift start
        if self.shifts and not self._is_in_shift(0.0):
            self.t = self._advance_to_next_shift_start(0.0)
            self.last_wip_time = self.t

        # KPI counters per stage
        self.stage_completed_counts: Dict[Any, int] = {s_id: 0 for s_id in self.stages}
        self.starvation_counts: Dict[Any, int] = {s_id: 0 for s_id in self.stages}
        self.blocking_counts: Dict[Any, int] = {s_id: 0 for s_id in self.stages}
        self.stage_defect_counts: Dict[Any, int] = {s_id: 0 for s_id in self.stages}
        # Order tracking for source stages (push mode)
        self.source_stage_orders: Dict[Any, int] = {
            s_id: 0 for s_id, s in self.stages.items() if not s.input_buffers
        }

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

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def enqueue_orders(self, qty: int = 1):
        """
        Increase WIP by 'qty' and trigger start attempts at all source stages
        (stages with NO input buffers). This mimics releasing orders into the system.
        In push mode, each source stage receives 'qty' orders to process.
        """
        # WIP area update before changing level
        self._accumulate_wip(self.t)
        self.current_wip += qty
        self.started += qty

        # For each 'source' stage (no input buffers), add orders and schedule ONE try_start
        for s in self.stages.values():
            if not s.input_buffers:
                self.source_stage_orders[s.stage_id] = self.source_stage_orders.get(s.stage_id, 0) + qty
                self._push_event(self.t, "try_start", {"stage_id": s.stage_id})

        self.log.append(f"{self._fmt_t()} Enqueued {qty} order(s) into source stages.")

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

    def run_until(self, t_stop: float, max_events: int = 1000000):
        """Run the simulation until time reaches t_stop or event cap is hit."""
        count = 0
        while self._queue and count < max_events:
            if self._queue[0].time > t_stop:
                break
            self.step()
            count += 1
        
        # Only fast-forward if we naturally reached t_stop (not hit max_events)
        if count < max_events or not self._queue:
            self._on_time_advance(t_stop)

    def run_for(self, dt: float, max_events: int = 100000):
        """Run the simulation for dt seconds from current time."""
        self.run_until(self.t + dt, max_events=max_events)

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
        service_level = self.finished / self.started
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

        for team_id, team in self.teams.items():
            # If currently busy, close interval temporally for utilization calculation
            if team.last_busy_start is not None:
                team.stop_busy(self.t)
                team.start_busy(self.t)
            utilization[team_id] = team.busy_time / sim_time
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
        }

    # --------------------------------------------------------------------------
    # Event handlers
    # --------------------------------------------------------------------------

    def _on_try_start(self, ev: Event):
        """
        Attempt to start a job at a stage:
        - Pulls exact items and quantities from configured input buffers.
        - Uses stage.required_materials directly (config-driven).
        - Handles starvation, blocking, and timing deterministically.
        """
        stage = self.stages.get(ev.payload.get("stage_id"))
        if stage is None:
            return

        # 1️⃣ Skip if already busy
        if stage.busy:
            self._push_event(self.t + 0.001, "try_start", {"stage_id": stage.stage_id})
            return

        # 2️⃣ Handle source stages (no inputs)
        if not stage.input_buffers:
            if self.source_stage_orders.get(stage.stage_id, 0) <= 0:
                return
            self.source_stage_orders[stage.stage_id] -= 1

        # 3️⃣ Read required materials from stage config
        required_materials = stage.required_materials or {}
        if not required_materials:
            self.log.append(f"{self._fmt_t()} '{stage.name}' has no required materials, skipping.")
            return

        pulled_items = []
        total_parts = 0

        # 4️⃣ Pull required materials from available buffers
        for item_id, required_qty in required_materials.items():
            pulled = False

            for b_id in stage.input_buffers:
                buf = self.buffers.get(b_id)
                if not buf:
                    continue

                if buf.can_pull_item(item_id, required_qty):
                    result = buf.pull_item(item_id, required_qty)
                    if result:
                        pulled_items.append((buf, item_id, required_qty))
                        total_parts += required_qty
                        pulled = True
                        break

            if not pulled:
                # rollback previously pulled items
                for prev_buf, prev_item, prev_qty in pulled_items:
                    prev_buf.push_item(prev_item, prev_qty)

                self.starvation_counts[stage.stage_id] += 1
                self.log.append(
                    f"{self._fmt_t()} '{stage.name}' waiting: insufficient '{item_id}' (need {required_qty})."
                )

                # retry later
                self._push_event(self.t + 0.5, "try_start", {"stage_id": stage.stage_id})
                return

        # 5️⃣ Engage team and mark stage busy
        team = self.teams.get(stage.team_id)
        if team:
            team.start_busy(self.t)

        stage.busy = True
        stage.output_ready = False
        stage.current_batch = pulled_items

        # 6️⃣ Compute deterministic processing time
        base_time = stage.base_process_time_sec
        total_work_time = base_time * max(1, total_parts)
        time_per_worker = total_work_time / max(1, stage.workers)
        ptime = sample_time(stage.time_distribution, time_per_worker)

        # 7️⃣ Optional random disruption (unchanged)
        missing_prob = float(self.random_events.get("missing_brick_prob", 0.0))
        if random.random() < missing_prob:
            penalty = float(self.random_events.get("missing_brick_penalty_sec", 5.0))
            ptime += penalty
            self.log.append(
                f"{self._fmt_t()} Disruption at '{stage.name}': missing bricks (+{penalty:.2f}s)."
            )

        # 8️⃣ Schedule job completion
        finish_t = self.t + ptime + float(stage.transport_time_sec or 0.0)
        self._push_event(finish_t, "complete", {"stage_id": stage.stage_id})

        # 9️⃣ Logging
        self.log.append(
            f"{self._fmt_t()} '{stage.name}' started job with {total_parts} items "
            f"(from {len(pulled_items)} types). ptime={ptime:.2f}s "
            f"(base={base_time:.2f}s, workers={stage.workers})."
        )

    def _on_complete(self, ev: Event):
        """
        Complete processing at a stage:
        - Handles defects with rework (1 retry max)
        - Deterministic push to defined output buffers
        - Avoids double defect checks on retry
        - Explicit blocking if outputs are full
        - Retrigger logic for source & non-source stages
        """
        stage = self.stages.get(ev.payload.get("stage_id"))
        if stage is None:
            return

        # 1️⃣ Release team (utilization ends)
        team = self.teams.get(stage.team_id)
        if team:
            team.stop_busy(self.t)

        # 2️⃣ Skip defect check if already output_ready (blocked retry)
        if getattr(stage, "output_ready", False):
            proceed_to_output = True
        else:
            proceed_to_output = True
            is_rework = ev.payload.get("is_rework", False)

            # 3️⃣ Defect handling (batch-level, with one-time rework)
            if stage.defect_rate and random.random() < stage.defect_rate:
                self.stage_defect_counts[stage.stage_id] = self.stage_defect_counts.get(stage.stage_id, 0) + 1

                if not is_rework:
                    # 🔁 First defect → schedule one rework
                    rework_time = sample_time(stage.time_distribution,
                                              stage.base_process_time_sec / max(1, stage.workers))
                    finish_t = self.t + rework_time + float(stage.transport_time_sec or 0.0)

                    self._push_event(
                        finish_t,
                        "complete",
                        {"stage_id": stage.stage_id, "is_rework": True}
                    )
                    self.log.append(
                        f"{self._fmt_t()} '{stage.name}' defect detected → rework scheduled (+{rework_time:.2f}s)."
                    )
                    return  # stop here — no push or free yet

                else:
                    # ⚠️ Already reworked once → scrap
                    self._accumulate_wip(self.t)
                    self.current_wip = max(0, self.current_wip - 1)
                    self.log.append(f"{self._fmt_t()} '{stage.name}' rework failed → scrapped batch.")
                    stage.busy = False
                    stage.blocked = False
                    self._push_event(self.t, "try_start", {"stage_id": stage.stage_id})
                    return

            # ✅ Mark batch as good (no defect)
            stage.output_ready = True

        # 4️⃣ Deterministic output push
        if proceed_to_output:
            all_push_success = True
            pushed_buffers = []

            for out_buffer_id, materials in stage.output_buffers.items():
                ob = self.buffers.get(out_buffer_id)

                if not ob:
                    self.log.append(f"{self._fmt_t()} '{stage.name}' error: missing output buffer '{out_buffer_id}'.")
                    all_push_success = False
                    continue

                for item_id, qty in materials.items():
                    success = ob.push_item(item_id, qty)

                    if success:
                        self.log.append(
                            f"{self._fmt_t()} '{stage.name}' pushed {qty}x '{item_id}' → '{out_buffer_id}'."
                        )
                        pushed_buffers.append(out_buffer_id)

                        if str(out_buffer_id) in [str(x) for x in self.finished_buffers]:
                            self.finished += qty
                            self.lead_times.append(self.t - 0.0)
                            self._accumulate_wip(self.t)
                            self.current_wip = max(0, self.current_wip - qty)
                            self.log.append(
                                f"{self._fmt_t()} Finished product(s) added to '{out_buffer_id}'. "
                                f"Total finished={self.finished}"
                            )
                    else:
                        # Buffer full → block and retry
                        self.blocking_counts[stage.stage_id] += 1
                        self.log.append(
                            f"{self._fmt_t()} '{stage.name}' output blocked: '{out_buffer_id}' full (waiting space)."
                        )

                        # Retry later, no defect recheck next time
                        self._push_event(self.t + 0.5, "complete", {"stage_id": stage.stage_id})
                        return

            # 5️⃣ Post-push housekeeping
            if all_push_success:
                stage.output_ready = False
                self.stage_completed_counts[stage.stage_id] += 1
                stage.blocked = False
                stage.busy = False

                # ✅ Re-trigger logic for source / non-source stages
                if not stage.input_buffers:
                    # Source stage: only if there are still orders
                    if self.source_stage_orders.get(stage.stage_id, 0) > 0:
                        self._push_event(self.t, "try_start", {"stage_id": stage.stage_id})
                else:
                    # Non-source: always re-trigger to keep pull alive
                    self._push_event(self.t, "try_start", {"stage_id": stage.stage_id})

                # ✅ Wake downstream consumers of pushed buffers
                for s in self.stages.values():
                    for b in pushed_buffers:
                        if b in s.input_buffers and not s.busy:
                            self._push_event(self.t, "try_start", {"stage_id": s.stage_id})

            else:
                # Still blocked, hold position
                stage.blocked = True
                stage.busy = True

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
        self._accumulate_wip(new_t)
        self.t = new_t

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _fmt_t(self) -> str:
        """Return current simulation time as a nicely formatted string."""
        return f"[{self.t:.2f}s]"

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
            snap[str(b_id)] = dict(buf.items)
        self.timeline.append(snap)
        self._last_sample_finished = self.finished
        self._last_sample_time = at_t


# ==============================================================================
# Built-in environment configuration (EDIT HERE)
# ==============================================================================

CONFIG: Dict[str, Any] = {
    # ----------------------------------------------------------------------------
    # Buffers (inventories). Use None capacity for "infinite" buffers.
    # A, B, C1, C2, C3, D1, D2, and E follow your LEGO flow notation.
    # ----------------------------------------------------------------------------
    "buffers": [
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
    {
        "buffer_id": "C1",
        "name": "C1 (Set for Axis Assembly)",
        "capacity": 9999,
        "initial_stock": {
            "bun01": 0
        }
    },
    {
        "buffer_id": "C2",
        "name": "C2 (Set for Chassis Assembly)",
        "capacity": 9999,
        "initial_stock": {
            "bun02": 0
        }
    },
    {
        "buffer_id": "C3",
        "name": "C3 (Final Assembly Only Parts)",
        "capacity": 9999,
        "initial_stock": {
            "bun03": 0
        }
    },
    {
        "buffer_id": "D1",
        "name": "D1 (Axis Subassembly)",
        "capacity": 9999,
        "initial_stock": {
            "saa01": 0, "saa02": 0
        }
    },
    {
        "buffer_id": "D2",
        "name": "D2 (Chassis Subassembly)",
        "capacity": 9999,
        "initial_stock": {
            "sac01": 0, "sac02": 0
        }
    },
    {
        "buffer_id": "E",
        "name": "E (Finished Gliders)",
        "capacity": 9999,
        "initial_stock": {
            "fg01": 0, "fg02": 0, "fg03": 0, "fg04": 0
        }
    }
    ],

    # ----------------------------------------------------------------------------
    # Teams (workers). One stage uses one team at a time.
    # ----------------------------------------------------------------------------
    "teams": [
        {"team_id": "T1", "name": "Type Sorting Team",  "size": 2, "shift_id": "day"},
        {"team_id": "T3", "name": "Axis Team",          "size": 2, "shift_id": "day"},
        {"team_id": "T4", "name": "Chassis Team",       "size": 2, "shift_id": "day"},
        {"team_id": "T5", "name": "Final Assembly Team","size": 3, "shift_id": "day"},
    ],

    # ----------------------------------------------------------------------------
    # Stages (process nodes).
    # - S1 (Type Sorting): no input buffers → pushes to B
    # - S2 (Set Sorting): pulls from B, probabilistic routing to C1/C2/C3
    # - S3 (Axis Assembly): C1 → D1
    # - S4 (Chassis Assembly): C2 → D2
    # - S5 (Final Assembly): D1 + D2 + C3 → E (multi-input)
    # ----------------------------------------------------------------------------
"stages": [
    {
        "stage_id": "S1",
        "name": "Type Sorting & Set Preparation",
        "team_id": "T1",
        "input_buffers": ["A"],
        "required_materials": {
            # parts taken from A for one full cycle of sorting
            # you can adjust quantities later depending on your balance
                "a01": 1, "a03": 2, "a05": 2, "a06": 1, "a07": 2,
                "b01": 1, "b02": 2, "b04": 1, "b05": 2, "b07": 1,
                "b09": 1, "b10": 2, "b11": 1, "b13": 1, "b14": 2,
                "b16": 1, "b18": 2,
                "c01": 1, "c02": 1, "c03": 1, "c04": 1, "c05": 4, "c07": 1, "c08": 1,
                "x01": 3, "x02": 4, "x03": 2
        },
        "output_buffers": {
            "C1": {'bun01': 1},   # Axis set
            "C2": {"bun02": 1},   # Chassis set
            "C3": {"bun03": 1}    # Final set
        },
        "base_process_time_sec": 3.0,
        "time_distribution": {"type": "triangular", "p1": 2.0, "p2": 3.0, "p3": 5.0},
        "transport_time_sec": 0.3,
        "defect_rate": 0.01,
        "rework_stage_id": "S1",    # if defect happens, rework same stage
        "workers": 2
    },
    {
        "stage_id": "S3",
        "name": "Axis Subassembly",
        "team_id": "T3",
        "input_buffers": ["C1"],
        "required_materials": {"bun01": 1},
        "output_buffers": {"D1": {"saa01": 1}},
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
        "required_materials": {"bun02": 1},
        "output_buffers": {"D2": {"sac01": 1}},
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
        "required_materials": {"bun03": 1, "saa01": 1, "sac01": 1},
        "output_buffers": {"E": {"fg01": 1}},
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
    # Global parameters (optional, free-form)
    # ----------------------------------------------------------------------------
    "parameters": {
        "target_takt_sec": 10.0,
        "timeline_sample_dt_sec": 5.0,
        "finished_buffer_ids": ["E"]
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
