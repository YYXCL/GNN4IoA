"""Versioned public interfaces. Inputs deliberately contain no outcome labels."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any
import numpy as np

SCHEMA_VERSION = "tc-mdsm/3"
GROUP_DIMS = {"A": 6, "C": 5, "E": 4, "R": 4}
METHODS = ("TC-MDSM", "GMATANN", "HM-MATAS", "Rule-Latency", "Full-State", "StaticGraph")


@dataclass
class ModelConfig:
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    history: int = 16
    top_b: int = 16
    gate_threshold: float = 0.5
    candidate_budgets: tuple[int, ...] = (32,)
    dropout: float = 0.05
    index: str = "exact"
    index_seed: int = 2718
    ablation: str = "none"
    prediction_mode: str = "residual"

    @property
    def state_dimension_budget(self):
        """State dimensions retained per group; distinct from candidate budget 32."""
        return self.top_b

    def __post_init__(self):
        if tuple(self.candidate_budgets)!=(32,):
            raise ValueError("v3 candidate budget must be exactly (32,)")
        if self.hidden % self.heads or self.top_b > self.hidden:
            raise ValueError("hidden must divide heads and top_b must not exceed hidden")
        if self.index not in ("exact", "lsh"):
            raise ValueError(self.index)


@dataclass
class Snapshot:
    time: float
    observed_at: list[float]
    states: dict[str, list]                 # [objects, history, features]
    missing: dict[str, list]
    skills: list[list[str]]
    tools: list[list[str]]
    knowledge: list[list[str]]
    descriptions: list[str]
    agent_compute: list[int]
    capacities: list[list[int]]             # GPU slots, RAM GiB, VRAM GiB
    links: list[list[int]]                  # physical endpoints: agents, then compute nodes
    bandwidth: list[int]                    # Mbps, fixed reservation per transfer
    available: list[float]
    reservations: list[dict] = field(default_factory=list)
    revision: str = ""
    agent_profiles: list[dict] = field(default_factory=list)
    schema: str = SCHEMA_VERSION

    def validate(self):
        if self.schema != SCHEMA_VERSION:
            raise ValueError("unsupported snapshot schema")
        if any(t > self.time for t in self.observed_at):
            raise ValueError("future observation is not a model input")
        if self.observed_at != sorted(self.observed_at):
            raise ValueError("history must be chronological")
        counts = {"A": len(self.skills), "R": len(self.skills), "C": len(self.capacities), "E": len(self.links)}
        for g, values in self.states.items():
            if len(values) != counts[g] or len(self.missing[g]) != counts[g]:
                raise ValueError(f"invalid {g} object axis")
            for history, missing in zip(values, self.missing[g]):
                if len(history) != len(self.observed_at) or len(missing) != len(history):
                    raise ValueError("invalid history axis")
                if any(len(row) != GROUP_DIMS[g] for row in history):
                    raise ValueError(f"invalid {g} feature axis")

    def feasible(self, task: Task) -> list[int]:
        return [i for i, skills in enumerate(self.skills)
                if task.skill in skills and set(task.tools) <= set(self.tools[i])
                and set(task.knowledge) <= set(self.knowledge[i])]

    def path(self, source: int, target: int) -> list[int] | None:
        if source == target:
            return []
        n = len(self.skills)
        a, b = self.agent_compute[source] + n, self.agent_compute[target] + n
        # Topology is immutable within a revision; invalidate when a topology update changes revision.
        cache_key=(id(self.links),len(self.links),self.revision)
        if getattr(self,"_path_cache_key",None)!=cache_key:
            self._path_lookup={tuple(sorted(edge)): i for i, edge in enumerate(self.links)}
            self._path_cache_key=cache_key
        lookup=self._path_lookup
        edges = [(source, a)] + ([] if a == b else [(a, b)]) + [(b, target)]
        result = [lookup.get(tuple(sorted(e))) for e in edges]
        return None if any(i is None for i in result) else result


@dataclass
class Task:
    id: int
    text: str
    skill: str
    tools: list[str]
    knowledge: list[str]
    work: float
    demand: list[int]


@dataclass
class Workflow:
    id: str
    tasks: list[Task]
    edges: list[list[float]]                # parent, child, MB
    release: float
    deadline: float                        # absolute time in seconds
    min_reliability: float = 0.9
    scenario: str = ""
    difficulty: str = ""
    template: str = ""

    def validate(self):
        if self.deadline <= self.release or not 0 < self.min_reliability <= 1:
            raise ValueError("invalid QoS")
        if [t.id for t in self.tasks] != list(range(len(self.tasks))):
            raise ValueError("task IDs must be consecutive topological indices")
        if any(not 0 <= a < b < len(self.tasks) or mb < 0 for a, b, mb in self.edges):
            raise ValueError("DAG edges must follow topological order")


@dataclass
class Observation:
    kind: str
    task: int
    agent: int
    duration: float
    success: bool
    started: float
    finished: float
    attempt: int
    propensity: float
    target_task: int = -1
    target_agent: int = -1
    failure_reason: str = ""


@dataclass
class Example:
    snapshot: Snapshot
    workflow: Workflow
    observations: list[Observation]
    provenance: dict[str, Any]
    schema: str = SCHEMA_VERSION

    def inputs(self):
        return self.snapshot, self.workflow

    def to_dict(self):
        def plain(value):
            if isinstance(value,np.ndarray):
                return value.tolist()
            if isinstance(value,dict):
                return {k:plain(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):
                return [plain(v) for v in value]
            return value
        return plain(asdict(self))

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != SCHEMA_VERSION:
            raise ValueError("legacy teacher labels require explicit conversion; they are not executions")
        w = dict(value["workflow"])
        w["tasks"] = [Task(**t) for t in w["tasks"]]
        result = cls(Snapshot(**value["snapshot"]), Workflow(**w),
                     [Observation(**o) for o in value["observations"]], value["provenance"])
        # Keep formal datasets in compact arrays instead of millions of boxed Python floats.
        result.snapshot.states={g:np.asarray(x,dtype=np.float32) for g,x in result.snapshot.states.items()}
        result.snapshot.missing={g:np.asarray(x,dtype=bool) for g,x in result.snapshot.missing.items()}
        result.snapshot.validate()
        result.workflow.validate()
        if any(o.started < result.snapshot.time or o.finished < o.started for o in result.observations):
            raise ValueError("execution labels must occur after the decision snapshot")
        return result


@dataclass
class Prediction:
    workflow_id: str
    candidates: dict[int, list[int]]
    nodes: dict[tuple[int, int], tuple[float, float]]
    edges: dict[tuple[int, int, int, int], tuple[float, float]]
    timing: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self):
        for duration, probability in list(self.nodes.values()) + list(self.edges.values()):
            if not math.isfinite(duration) or duration < 0 or not 0 < probability <= 1:
                raise ValueError("invalid prediction")


@dataclass
class Plan:
    assignments: list[dict]
    transfers: list[dict]
    admitted: dict[str, bool]
    status: str
    elapsed: float
    reasons: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
