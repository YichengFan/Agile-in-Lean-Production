
# env.py
# LEGO Lean Production Simulation Environment (Discrete-Event)
# --------------------------------------------------------------------------------------
# This module implements a small, dependency-light discrete-event simulation (DES)
# environment tailored to the LEGO lean production project.
# - Compatible with a dict-like config (see SIM_CONFIG schema in sim_config_auto.py).
# - Focus on clarity, testability, and "click to play" step-wise execution.
# --------------------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import heapq
import random
import math
import time

# ---------------------------------------
# Utilities
# ---------------------------------------

def now_ms() -> int:
    """Wall-clock milliseconds (for profiling, not simulation time)."""
    return int(time.time() * 1000)

def sample_time(dist: Dict[str, Any], base: float) -> float:
    """Sample a processing time given a distribution descriptor.
    Supported types: constant, normal, lognormal, triangular, uniform, exponential.
    Fallback is constant.
    """
    if not dist:
        return max(0.0, float(base))
    t = (dist.get('type') or 'constant').lower()
    p1, p2, p3 = dist.get('p1'), dist.get('p2'), dist.get('p3')
    if t == 'constant':
        return max(0.0, float(base))
    if t == 'normal':
        mu = float(p1) if p1 is not None else float(base)
        sigma = float(p2) if p2 is not None else 0.1 * mu
        return max(0.0, random.gauss(mu, sigma))
    if t == 'lognormal':
        mu = float(p1) if p1 is not None else math.log(max(1e-6, base))
        sigma = float(p2) if p2 is not None else 0.25
        return max(0.0, random.lognormvariate(mu, sigma))
    if t == 'triangular':
        low = float(p1) if p1 is not None else 0.5 * base
        mode = float(p2) if p2 is not None else base
        high = float(p3) if p3 is not None else 1.5 * base
        return max(0.0, random.triangular(low, high, mode))
    if t == 'uniform':
        a = float(p1) if p1 is not None else 0.8 * base
        b = float(p2) if p2 is not None else 1.2 * base
        return max(0.0, random.uniform(a, b))
    if t == 'exponential':
        lam = 1.0 / float(base) if base else 1.0
        return max(0.0, random.expovariate(lam))
    # Fallback
    return max(0.0, float(base))

# ---------------------------------------
# Data structures
# ---------------------------------------

@dataclass
class Buffer:
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
    team_id: Any
    name: str
    size: int = 1
    shift_id: Optional[Any] = None

    # utilization tracking
    busy_time: float = 0.0  # simulated minutes/seconds depending on base unit
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
    stage_id: Any
    name: str
    team_id: Any
    input_buffer: Optional[Any] = None
    output_buffer: Optional[Any] = None
    base_process_time_sec: float = 1.0
    time_distribution: Dict[str, Any] = field(default_factory=dict)
    transport_time_sec: float = 0.0
    defect_rate: float = 0.0
    rework_stage_id: Optional[Any] = None

    # internal
    busy: bool = False

@dataclass(order=True)
class Event:
    time: float
    seq: int
    kind: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)

# ---------------------------------------
# Simulation environment
# ---------------------------------------

class LegoLeanEnv:
    """Discrete-event simulation environment for the LEGO lean production flow."""

    def __init__(self, config: Dict[str, Any], time_unit: str = "sec", seed: Optional[int] = None):
        self.cfg = config
        self.time_unit = time_unit  # 'sec' by default
        if seed is not None:
            random.seed(seed)

        # Build objects
        self.buffers: Dict[Any, Buffer] = {}
        for b in self.cfg.get('buffers', []):
            self.buffers[b.get('buffer_id') or b.get('name')] = Buffer(
                buffer_id=b.get('buffer_id') or b.get('name'),
                name=b.get('name') or str(b.get('buffer_id')),
                capacity=b.get('capacity'),
                initial_stock=b.get('initial_stock') or 0
            )

        self.teams: Dict[Any, Team] = {}
        for t in self.cfg.get('teams', []):
            self.teams[t.get('team_id') or t.get('name')] = Team(
                team_id=t.get('team_id') or t.get('name'),
                name=t.get('name') or str(t.get('team_id')),
                size=int(t.get('size') or 1),
                shift_id=t.get('shift_id')
            )

        self.stages: Dict[Any, Stage] = {}
        for s in self.cfg.get('stages', []):
            self.stages[s.get('stage_id') or s.get('name')] = Stage(
                stage_id=s.get('stage_id') or s.get('name'),
                name=s.get('name') or str(s.get('stage_id')),
                team_id=s.get('team_id'),
                input_buffer=s.get('input_buffer'),
                output_buffer=s.get('output_buffer'),
                base_process_time_sec=float(s.get('base_process_time_sec') or 1.0),
                time_distribution=s.get('time_distribution') or {},
                transport_time_sec=float(s.get('transport_time_sec') or 0.0),
                defect_rate=float(s.get('defect_rate') or 0.0),
                rework_stage_id=s.get('rework_stage_id')
            )

        # Shifts: optional, list of dicts
        self.shifts = self.cfg.get('shift_schedule', [])

        # Parameters & random events
        self.parameters = self.cfg.get('parameters', {})
        self.random_events = self.cfg.get('random_events', {})

        # Event queue
        self._evt_seq = 0
        self._queue: List[Event] = []

        # Simulation time
        self.t: float = 0.0  # seconds by default

        # KPIs
        self.finished: int = 0
        self.started: int = 0
        self.lead_times: List[float] = []
        self.wip_time_area: float = 0.0
        self.last_wip_time: float = 0.0
        self.current_wip: int = 0

        # Trace / logs
        self.log: List[str] = []

    # -------------- Shift logic --------------

    def _is_in_shift(self, t: float) -> bool:
        if not self.shifts:
            return True
        minute_in_day = (t / 60.0) % 1440.0
        for sh in self.shifts:
            start_min = float(sh.get('start_minute', 0))
            end_min = float(sh.get('end_minute', 1440))
            if start_min <= minute_in_day <= end_min:
                return True
        return False

    def _advance_to_next_shift_start(self, t: float) -> float:
        if not self.shifts:
            return t
        minute_in_day = (t / 60.0) % 1440.0
        starts = sorted([float(s.get('start_minute', 0)) for s in self.shifts])
        for st in starts:
            if st > minute_in_day:
                delta_min = st - minute_in_day
                return t + delta_min * 60.0
        delta_min = (1440.0 - minute_in_day) + starts[0]
        return t + delta_min * 60.0

    # -------------- Event queue helpers --------------

    def _push_event(self, when: float, kind: str, payload: Optional[Dict[str, Any]] = None):
        self._evt_seq += 1
        heapq.heappush(self._queue, Event(when, self._evt_seq, kind, payload or {}))

    def _pop_event(self) -> Optional[Event]:
        if not self._queue:
            return None
        return heapq.heappop(self._queue)

    # -------------- Public API --------------

    def enqueue_order(self, qty: int = 1, start_buffer: Optional[Any] = None):
        """Add work into the flow. If a start_buffer is supplied, push units there; else
        treat it as "virtual" release starting at first stage without input buffer."""
        self._accumulate_wip(self.t)
        self.current_wip += qty
        self.started += qty

        if start_buffer is not None and start_buffer in self.buffers:
            self.buffers[start_buffer].push(qty)
            self.log.append(f"{self._fmt_t()} Enqueue {qty} unit(s) into buffer '{start_buffer}'.")
        else:
            for stage in self.stages.values():
                if not stage.input_buffer:
                    self._push_event(self.t, 'try_start', {'stage_id': stage.stage_id, 'qty': qty})
                    self.log.append(f"{self._fmt_t()} Enqueue {qty} unit(s) to stage '{stage.name}' (no input buffer).")

    def step(self) -> Optional[Event]:
        """Process the next scheduled event. Returns the processed Event or None if done."""
        ev = self._pop_event()
        if ev is None:
            return None

        if not self._is_in_shift(ev.time):
            shifted = self._advance_to_next_shift_start(ev.time)
            self._push_event(shifted, ev.kind, ev.payload)
            return self._pop_event()

        self._on_time_advance(ev.time)

        handler = getattr(self, f"_on_{ev.kind}", None)
        if handler:
            handler(ev)
        else:
            self.log.append(f"{self._fmt_t()} [WARN] No handler for event kind='{ev.kind}'.")
        return ev

    def run_until(self, t_stop: float, max_events: int = 100000):
        """Run until simulation time reaches t_stop or event cap."""
        count = 0
        while self._queue and count < max_events:
            if self._queue[0].time > t_stop:
                break
            self.step()
            count += 1
        self._on_time_advance(t_stop)

    def run_for(self, dt: float, max_events: int = 100000):
        self.run_until(self.t + dt, max_events=max_events)

    # -------------- KPI computation --------------

    def _accumulate_wip(self, new_t: float):
        dt = max(0.0, new_t - self.last_wip_time)
        self.wip_time_area += self.current_wip * dt
        self.last_wip_time = new_t

    def get_kpis(self) -> Dict[str, Any]:
        sim_time = max(1e-9, self.t)
        throughput = self.finished / sim_time
        lead_time_avg = sum(self.lead_times) / len(self.lead_times) if self.lead_times else 0.0
        wip_avg = self.wip_time_area / sim_time
        util = {}
        for team_id, team in self.teams.items():
            if team.last_busy_start is not None:
                team.stop_busy(self.t)
                team.start_busy(self.t)
            util[team_id] = team.busy_time / sim_time
        return {
            'sim_time': sim_time,
            'throughput_per_sec': throughput,
            'lead_time_avg': lead_time_avg,
            'wip_avg': wip_avg,
            'utilization': util,
            'finished': self.finished,
            'started': self.started
        }

    # -------------- Event Handlers --------------

    def _on_try_start(self, ev: Event):
        stage = self.stages.get(ev.payload.get('stage_id'))
        if stage is None:
            return

        if stage.busy:
            self._push_event(self.t + 0.001, 'try_start', {'stage_id': stage.stage_id})
            return

        if stage.input_buffer:
            buf = self.buffers.get(stage.input_buffer)
            if not buf or not buf.pull(1):
                self._push_event(self.t + 0.5, 'try_start', {'stage_id': stage.stage_id})
                return

        team = self.teams.get(stage.team_id)
        if team:
            team.start_busy(self.t)

        stage.busy = True

        ptime = sample_time(stage.time_distribution, stage.base_process_time_sec)
        missing_prob = float(self.random_events.get('missing_brick_prob', 0.0))
        if random.random() < missing_prob:
            penalty = float(self.random_events.get('missing_brick_penalty_sec', 5.0))
            ptime += penalty
            self.log.append(f"{self._fmt_t()} Disruption at '{stage.name}': missing bricks (+{penalty:.2f}s).")

        done_t = self.t + ptime + float(stage.transport_time_sec or 0.0)
        self._push_event(done_t, 'complete', {'stage_id': stage.stage_id})

    def _on_complete(self, ev: Event):
        stage = self.stages.get(ev.payload.get('stage_id'))
        if stage is None:
            return

        team = self.teams.get(stage.team_id)
        if team:
            team.stop_busy(self.t)

        go_to_output = True
        if stage.defect_rate and random.random() < stage.defect_rate:
            if stage.rework_stage_id and stage.rework_stage_id in self.stages:
                self._push_event(self.t, 'try_start', {'stage_id': stage.rework_stage_id})
                go_to_output = False
                self.log.append(f"{self._fmt_t()} '{stage.name}' defect → rework at '{stage.rework_stage_id}'.")
            else:
                self._accumulate_wip(self.t)
                self.current_wip = max(0, self.current_wip - 1)
                go_to_output = False
                self.log.append(f"{self._fmt_t()} '{stage.name}' defect → scrapped item.")

        if go_to_output and stage.output_buffer:
            ob = self.buffers.get(stage.output_buffer)
            if ob and ob.push(1):
                self.log.append(f"{self._fmt_t()} '{stage.name}' pushed item to buffer '{stage.output_buffer}'.")
            else:
                self._push_event(self.t + 0.5, 'complete', {'stage_id': stage.stage_id})
                return

        if go_to_output and not stage.output_buffer:
            self.finished += 1
            self.lead_times.append(self.t - 0.0)
            self._accumulate_wip(self.t)
            self.current_wip = max(0, self.current_wip - 1)
            self.log.append(f"{self._fmt_t()} Product finished at stage '{stage.name}'. Total finished = {self.finished}.")

        stage.busy = False
        self._push_event(self.t, 'try_start', {'stage_id': stage.stage_id})

        if stage.output_buffer:
            for s in self.stages.values():
                if s.input_buffer == stage.output_buffer:
                    self._push_event(self.t, 'try_start', {'stage_id': s.stage_id})

    # -------------- Internal time advance --------------

    def _on_time_advance(self, new_t: float):
        new_t = max(new_t, self.t)
        if new_t == self.t:
            return
        self._accumulate_wip(new_t)
        self.t = new_t

    # -------------- Helpers --------------

    def _fmt_t(self) -> str:
        return f"[t={self.t:.2f}{self.time_unit}]"

# ---------------------------------------
# Example usage (can be removed in production)
# ---------------------------------------

if __name__ == '__main__':
    try:
        from sim_config_auto import SIM_CONFIG
        cfg = SIM_CONFIG
    except Exception:
        cfg = {
            "buffers": [
                {"buffer_id": "B", "name": "B", "capacity": 999, "initial_stock": 5},
                {"buffer_id": "E", "name": "E", "capacity": 999, "initial_stock": 0},
            ],
            "teams": [
                {"team_id": "T1", "name": "Axis Team", "size": 1},
            ],
            "stages": [
                {
                    "stage_id": "S1",
                    "name": "Axis Assembly",
                    "team_id": "T1",
                    "input_buffer": "B",
                    "output_buffer": "E",
                    "base_process_time_sec": 3.0,
                    "time_distribution": {"type": "triangular", "p1": 2.0, "p2": 3.0, "p3": 5.0},
                    "transport_time_sec": 0.5,
                    "defect_rate": 0.05
                }
            ],
            "parameters": {},
            "random_events": {"missing_brick_prob": 0.10, "missing_brick_penalty_sec": 2.0},
            "shift_schedule": [{"shift_id": "day", "start_minute": 0, "end_minute": 480}]
        }

    env = LegoLeanEnv(cfg, time_unit="sec", seed=42)
    for _ in range(10):
        env.enqueue_order(qty=1, start_buffer="B")
    for s in env.stages.values():
        env._push_event(env.t, 'try_start', {'stage_id': s.stage_id})
    env.run_for(300.0)

    print("--- KPIs ---")
    for k, v in env.get_kpis().items():
        print(k, ":", v)

    print("\n--- Trace (last 10) ---")
    for line in env.log[-10:]:
        print(line)
