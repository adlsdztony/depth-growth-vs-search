"""Parent-started monotonic allocation clock; no simulator or GPU dependencies.

The coordinator must capture ``allocation_start_ns`` BEFORE Popen and pass both
absolute timestamps to the child. Linux CLOCK_MONOTONIC timestamps are comparable
between processes on the same boot. This helper cannot interrupt blocking work;
the coordinator must enforce process-group termination and collect cleanup costs.
"""
from dataclasses import dataclass, field
import math
import time
from typing import Callable


class BudgetExpired(RuntimeError):
    """A paid allocation stage has no remaining time."""


def _seconds(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _ns(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class AllocationClock:
    allocation_start_ns: int
    deadline_ns: int
    final_assessment_reserve_s: float
    cleanup_reserve_s: float
    now_ns: Callable[[], int] = field(default=time.monotonic_ns, repr=False, compare=False)

    def __post_init__(self):
        start = _ns(self.allocation_start_ns, "allocation_start_ns")
        end = _ns(self.deadline_ns, "deadline_ns")
        if end <= start:
            raise ValueError("deadline must be later than allocation start")
        assessment = _seconds(self.final_assessment_reserve_s, "assessment reserve")
        cleanup = _seconds(self.cleanup_reserve_s, "cleanup reserve")
        object.__setattr__(self, "final_assessment_reserve_s", assessment)
        object.__setattr__(self, "cleanup_reserve_s", cleanup)
        if assessment + cleanup >= (end - start) / 1e9:
            raise ValueError("reserves must leave a positive training allowance")
        if not callable(self.now_ns):
            raise ValueError("now_ns must be callable")

    def stage_deadline_ns(self, stage="train"):
        if stage == "train":
            return self.deadline_ns - math.ceil(1e9 * (self.final_assessment_reserve_s + self.cleanup_reserve_s))
        if stage == "assessment":
            return self.deadline_ns - math.ceil(1e9 * self.cleanup_reserve_s)
        if stage == "cleanup":
            return self.deadline_ns
        raise ValueError(f"unknown allocation stage: {stage}")

    def _now(self):
        now = _ns(self.now_ns(), "current monotonic timestamp")
        if now < self.allocation_start_ns:
            raise ValueError("current clock precedes allocation start; wrong boot/clock domain")
        return now

    @property
    def elapsed_s(self):
        return (self._now() - self.allocation_start_ns) / 1e9

    @property
    def remaining_s(self):
        return max(0.0, (self.deadline_ns - self._now()) / 1e9)

    @property
    def train_remaining_s(self):
        return self.stage_remaining_s("train")

    def stage_remaining_s(self, stage="train"):
        end = self.stage_deadline_ns(stage)
        return max(0.0, (end - self._now()) / 1e9)

    def require(self, stage="train"):
        end = self.stage_deadline_ns(stage)
        now = self._now()
        if now >= end:
            raise BudgetExpired(f"{stage} deadline reached; overrun_s={(now-end)/1e9:.9f}")

    def can_start(self, stage="train", estimated_s=0.0):
        """Allow only estimates strictly inside a stage; estimates are not guarantees."""
        estimate = _seconds(estimated_s, "estimated duration")
        end = self.stage_deadline_ns(stage)
        return self._now() + math.ceil(estimate * 1e9) < end

    def report(self):
        """One clock sample, including overruns rather than hiding them with clipping."""
        now = self._now()
        return {
            "allocation_start_ns": self.allocation_start_ns,
            "deadline_ns": self.deadline_ns,
            "sampled_monotonic_ns": now,
            "allocated_s": (self.deadline_ns-self.allocation_start_ns)/1e9,
            "elapsed_s": (now-self.allocation_start_ns)/1e9,
            "remaining_s": max(0.0, (self.deadline_ns-now)/1e9),
            "overrun_s": max(0.0, (now-self.deadline_ns)/1e9),
            "train_deadline_ns": self.stage_deadline_ns("train"),
            "assessment_deadline_ns": self.stage_deadline_ns("assessment"),
            "final_assessment_reserve_s": self.final_assessment_reserve_s,
            "cleanup_reserve_s": self.cleanup_reserve_s,
            "time_basis": "parent-started CLOCK_MONOTONIC; includes imports/setup through cleanup",
            "enforcement_limit": "cooperative stage checks; coordinator must bound blocking work/process cleanup",
        }


def worker_affinity_receipt(allowed_cpus, logical_core_limit=12):
    """Verify current affinity; descendants inherit it unless they explicitly escape.

    This does not pin, grant permissions, measure CPU utilization, enforce cgroup
    limits, or prove all descendants retained affinity. The coordinator must check
    descendant scope. Restricting affinity is a resource ceiling, not equal CPU use.
    """
    import os
    allowed = set(allowed_cpus)
    if not allowed or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in allowed):
        raise ValueError("allowed_cpus must contain nonnegative CPU indices")
    if isinstance(logical_core_limit, bool) or not isinstance(logical_core_limit, int) or logical_core_limit < 1:
        raise ValueError("logical_core_limit must be a positive integer")
    if len(allowed) > logical_core_limit:
        raise ValueError("allowed CPU set exceeds logical-core limit")
    actual = set(os.sched_getaffinity(0))
    if not actual or not actual <= allowed:
        raise RuntimeError(f"worker affinity {sorted(actual)} escapes allowed CPUs {sorted(allowed)}")
    return {"allowed_cpus": sorted(allowed), "actual_cpus": sorted(actual),
            "logical_core_limit": logical_core_limit,
            "scope": "current process affinity only; no equal CPU-time or descendant proof"}
