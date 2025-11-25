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
    """Finite (or infinite if capacity=None) storage for parts/products."""
    buffer_id: Any
    name: str
    capacity: Optional[int] = None
    initial_stock: int = 0

    # internal state
    current: int = 0

    def __post_init__(self):
        self.current = int(self.initial_stock or 0)

    def can_pull(self, qty: int = 1) -> bool:
        return self.current >= qty

    def pull(self, qty: int = 1) -> bool:
        if self.can_pull(qty):
            self.current -= qty
            return True
        return False

    def can_push(self, qty: int = 1) -> bool:
        if self.capacity is None:
            return True
        return (self.current + qty) <= int(self.capacity)

    def push(self, qty: int = 1) -> bool:
        if self.can_push(qty):
            self.current += qty
            return True
        return False


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
    Process node.
    - Supports multiple input buffers (require 1 unit from each before starting).
    - Supports either: a single output buffer, or probabilistic routing to one of many outputs.
      Use 'output_buffer' (str) OR 'output_rules' (list of {'buffer_id': str, 'p': float}).
    - workers: number of workers at this stage. Processing time = base_time / workers
    """
    stage_id: Any
    name: str
    team_id: Any
    input_buffers: List[Any] = field(default_factory=list)  # e.g., ['D1', 'D2', 'C3'] for Final Assembly
    output_buffer: Optional[Any] = None                     # single deterministic output
    output_rules: Optional[List[Dict[str, Any]]] = None     # probabilistic outputs [{'buffer_id': 'C1', 'p': 0.4}, ...]

    base_process_time_sec: float = 1.0
    time_distribution: Dict[str, Any] = field(default_factory=dict)
    transport_time_sec: float = 0.0
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
                output_buffer=s.get("output_buffer"),
                output_rules=s.get("output_rules"),
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
        """Attempt to start a job at the stage, pulling inputs and engaging the team."""
        stage = self.stages.get(ev.payload.get("stage_id"))
        if stage is None:
            return

        # If stage is busy, try again shortly
        if stage.busy:
            self._push_event(self.t + 0.001, "try_start", {"stage_id": stage.stage_id})
            return

        # For source stages (no input buffers), check if there are orders to process
        if not stage.input_buffers:
            if self.source_stage_orders.get(stage.stage_id, 0) <= 0:
                # No orders to process, don't start
                return
            # Consume one order
            self.source_stage_orders[stage.stage_id] -= 1

        # Check all required inputs (multi-input allowed)
        pulled_inputs: List[Buffer] = []
        for b_id in stage.input_buffers:
            buf = self.buffers.get(b_id)
            if not buf or not buf.pull(1):
                # roll back any inputs we might have pulled already
                for pb in pulled_inputs:
                    pb.push(1)
                # retry later
                self.starvation_counts[stage.stage_id] += 1
                self.log.append(f"{self._fmt_t()} '{stage.name}' waiting: insufficient '{b_id}'.")
                self._push_event(self.t + 0.5, "try_start", {"stage_id": stage.stage_id})
                return
            pulled_inputs.append(buf)

        # Engage team (utilization starts)
        team = self.teams.get(stage.team_id)
        if team:
            team.start_busy(self.t)

        stage.busy = True

        # Draw processing time from distribution + optional disruption penalty
        # base_process_time_sec is "per worker", so divide by number of workers
        base_time_per_unit = stage.base_process_time_sec / max(1, stage.workers)
        ptime = sample_time(stage.time_distribution, base_time_per_unit)

        # Random disruption: missing bricks → extra processing time penalty
        missing_prob = float(self.random_events.get("missing_brick_prob", 0.0))
        if random.random() < missing_prob:
            penalty = float(self.random_events.get("missing_brick_penalty_sec", 5.0))
            ptime += penalty
            self.log.append(f"{self._fmt_t()} Disruption at '{stage.name}': missing bricks (+{penalty:.2f}s).")

        finish_t = self.t + ptime + float(stage.transport_time_sec or 0.0)
        self._push_event(finish_t, "complete", {"stage_id": stage.stage_id})

    def _on_complete(self, ev: Event):
        """Complete processing at a stage, handle defects/rework, then push outputs."""
        stage = self.stages.get(ev.payload.get("stage_id"))
        if stage is None:
            return

        # Release team (utilization ends)
        team = self.teams.get(stage.team_id)
        if team:
            team.stop_busy(self.t)

        # Defect handling (rework or scrap)
        proceed_to_output = True
        if stage.defect_rate and random.random() < stage.defect_rate:
            """Defect counts logic"""
            self.stage_defect_counts[stage.stage_id] = self.stage_defect_counts.get(stage.stage_id, 0) + 1
            if stage.rework_stage_id and stage.rework_stage_id in self.stages:
                self._push_event(self.t, "try_start", {"stage_id": stage.rework_stage_id})
                proceed_to_output = False
                self.log.append(f"{self._fmt_t()} '{stage.name}' defect → rework at '{stage.rework_stage_id}'.")
            else:
                # Scrap: reduce WIP (this item leaves the system)
                self._accumulate_wip(self.t)
                self.current_wip = max(0, self.current_wip - 1)
                proceed_to_output = False
                self.log.append(f"{self._fmt_t()} '{stage.name}' defect → scrapped item.")

        chosen_out = None
        if proceed_to_output:
            # Determine output buffer (single deterministic OR probabilistic rules)
            out_buffer_id = stage.output_buffer
            if stage.output_rules:
                if self.routing_mode == "deterministic":
                    rules = stage.output_rules
                    if stage.stage_id not in self._served_counts:
                        self._served_counts[stage.stage_id] = {}
                    if stage.stage_id not in self._rr_index:
                        self._rr_index[stage.stage_id] = 0
                    served = self._served_counts[stage.stage_id]
                    # Compute balance score: served / p. Lower is more under-served.
                    scores = []
                    for idx, rule in enumerate(rules):
                        buf_id = str(rule.get("buffer_id"))
                        p = max(1e-9, float(rule.get("p", 0.0)))
                        c = float(served.get(buf_id, 0))
                        score = c / p
                        scores.append((score, idx, buf_id))
                    scores.sort(key=lambda x: x[0])
                    # Find all with minimal score (tie set)
                    min_score = scores[0][0]
                    tie = [t for t in scores if abs(t[0] - min_score) <= 1e-12]
                    if len(tie) == 1:
                        out_buffer_id = tie[0][2]
                    else:
                        # Round-robin among ties to avoid always picking first rule
                        rr = self._rr_index[stage.stage_id] % len(tie)
                        out_buffer_id = tie[rr][2]
                        self._rr_index[stage.stage_id] = (self._rr_index[stage.stage_id] + 1) % len(tie)
                else:
                    r = random.random()
                    cum = 0.0
                    chosen = None
                    for rule in stage.output_rules:
                        cum += float(rule.get("p", 0.0))
                        if r <= cum:
                            chosen = rule.get("buffer_id")
                            break
                    out_buffer_id = chosen or out_buffer_id

            if out_buffer_id:
                ob = self.buffers.get(out_buffer_id)
                if ob and ob.push(1):
                    self.log.append(f"{self._fmt_t()} '{stage.name}' pushed item → '{out_buffer_id}'.")
                    chosen_out = out_buffer_id
                    # Count as finished if pushed into a configured finished buffer
                    if str(out_buffer_id) in [str(x) for x in self.finished_buffers]:
                        self.finished += 1
                        self.lead_times.append(self.t - 0.0)
                        self._accumulate_wip(self.t)
                        self.current_wip = max(0, self.current_wip - 1)
                        self.log.append(f"{self._fmt_t()} Product finished into buffer '{out_buffer_id}'. Finished={self.finished}")
                    # Update per-stage completion counters and served counts
                    self.stage_completed_counts[stage.stage_id] += 1
                    if self.routing_mode == "deterministic" and stage.output_rules:
                        sc = self._served_counts.setdefault(stage.stage_id, {})
                        key = str(out_buffer_id)
                        sc[key] = sc.get(key, 0) + 1
                else:
                    # Output buffer full or missing → delay and retry the completion
                    self.blocking_counts[stage.stage_id] += 1
                    self._push_event(self.t + 0.5, "complete", {"stage_id": stage.stage_id})
                    return
            else:
                # No output buffer means this is a sink/final stage → finished product
                self.finished += 1
                # Simple lead-time approximation: we assume each order "started at t=0".
                # If you track per-unit start times, record exact lead times here.
                self.lead_times.append(self.t - 0.0)
                self._accumulate_wip(self.t)
                self.current_wip = max(0, self.current_wip - 1)
                self.log.append(f"{self._fmt_t()} Product finished at '{stage.name}'. Finished={self.finished}")
                self.stage_completed_counts[stage.stage_id] += 1

        # Free the stage and immediately attempt next start at this stage
        stage.busy = False
        
        # For source stages (no inputs), only re-trigger if there are orders
        # For non-source stages, always re-trigger (they pull from buffers)
        if not stage.input_buffers:
            # Source stage: check order count
            if self.source_stage_orders.get(stage.stage_id, 0) > 0:
                self._push_event(self.t, "try_start", {"stage_id": stage.stage_id})
        else:
            # Non-source: always try again
            self._push_event(self.t, "try_start", {"stage_id": stage.stage_id})

        # Only wake consumers of the actual chosen output buffer (if they're not busy)
        if chosen_out:
            for s in self.stages.values():
                if chosen_out in s.input_buffers and not s.busy:
                    self._push_event(self.t, "try_start", {"stage_id": s.stage_id})

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
            snap[str(b_id)] = int(buf.current)
        self.timeline.append(snap)
        self._last_sample_finished = self.finished
        self._last_sample_time = at_t


# ==============================================================================
# Built-in environment configuration (EDIT HERE)
# ==============================================================================

CONFIG: Dict[str, Any] = {
    # ----------------------------------------------------------------------------
    # Buffers (inventories). Use None capacity for "infinite" buffers.
    # B, C1, C2, C3, D1, D2, and E follow your LEGO flow notation.
    # ----------------------------------------------------------------------------
    "buffers": [
        {"buffer_id": "B",  "name": "Warehouse B (post-TypeSorting)", "capacity": 999, "initial_stock": 30},
        {"buffer_id": "C1", "name": "C1 (Axis Parts)",                "capacity": 999, "initial_stock": 0},
        {"buffer_id": "C2", "name": "C2 (Chassis Parts)",             "capacity": 999, "initial_stock": 0},
        {"buffer_id": "C3", "name": "C3 (Final Assembly Only Parts)", "capacity": 999, "initial_stock": 0},
        {"buffer_id": "D1", "name": "D1 (Axis Subassembly)",          "capacity": 999, "initial_stock": 0},
        {"buffer_id": "D2", "name": "D2 (Chassis Subassembly)",       "capacity": 999, "initial_stock": 0},
        {"buffer_id": "E",  "name": "E (Finished Gliders)",           "capacity": 999, "initial_stock": 0},
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
            "name": "Type Sorting",
            "team_id": "T1",
            "input_buffers": [],               # source stage (no inputs)
            "output_buffer": "B",              # deterministic output
            "base_process_time_sec": 2.5,      # time per unit per worker
            "time_distribution": {"type": "triangular", "p1": 2.0, "p2": 2.5, "p3": 4.0},
            "transport_time_sec": 0.2,
            "defect_rate": 0.00,
            "workers": 2                       # number of workers at this stage
        },
        {
            "stage_id": "S2",
            "name": "Set Sorting",
            "team_id": "T2",
            "input_buffers": ["B"],
            "output_rules": [                   # probabilistic split into C1/C2/C3
                {"buffer_id": "C1", "p": 0.40},
                {"buffer_id": "C2", "p": 0.40},
                {"buffer_id": "C3", "p": 0.20}
            ],
            "base_process_time_sec": 3.0,      # time per unit per worker
            "time_distribution": {"type": "triangular", "p1": 2.0, "p2": 3.0, "p3": 5.0},
            "transport_time_sec": 0.3,
            "defect_rate": 0.01,
            "rework_stage_id": "S2",           # simple rework back to self
            "workers": 2                       # number of workers
        },
        {
            "stage_id": "S3",
            "name": "Axis Assembly",
            "team_id": "T3",
            "input_buffers": ["C1"],
            "output_buffer": "D1",
            "base_process_time_sec": 4.0,      # time per unit per worker
            "time_distribution": {"type": "triangular", "p1": 3.0, "p2": 4.0, "p3": 6.0},
            "transport_time_sec": 0.4,
            "defect_rate": 0.02,
            "workers": 2                       # number of workers
        },
        {
            "stage_id": "S4",
            "name": "Chassis Assembly",
            "team_id": "T4",
            "input_buffers": ["C2"],
            "output_buffer": "D2",
            "base_process_time_sec": 4.0,      # time per unit per worker
            "time_distribution": {"type": "triangular", "p1": 3.0, "p2": 4.0, "p3": 6.0},
            "transport_time_sec": 0.4,
            "defect_rate": 0.02,
            "workers": 2                       # number of workers
        },
        {
            "stage_id": "S5",
            "name": "Final Assembly",
            "team_id": "T5",
            "input_buffers": ["D1", "D2", "C3"],   # multi-input
            "output_buffer": "E",
            "base_process_time_sec": 6.0,      # time per unit per worker
            "time_distribution": {"type": "triangular", "p1": 5.0, "p2": 6.0, "p3": 9.0},
            "transport_time_sec": 0.5,
            "defect_rate": 0.03,
            "workers": 3                       # number of workers
        },
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
