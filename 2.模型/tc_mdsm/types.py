"""第0步(数据定义阶段)：带版本号的公共接口定义。模型输入刻意不含任何结果标签（避免标签泄漏）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any
import numpy as np

SCHEMA_VERSION = "tc-mdsm/3"
# 四类异构图对象的单步状态特征维度：
# A=智能体(Agent)，C=算力节点(Compute)，E=网络边(Edge)，R=可靠性(Reliability)
GROUP_DIMS = {"A": 6, "C": 5, "E": 4, "R": 4}
# 支持的模型与对比基线方法名
METHODS = ("TC-MDSM", "GMATANN", "HM-MATAS", "Rule-Latency", "Full-State", "StaticGraph")


@dataclass
class ModelConfig:
    """模型超参数配置。"""
    hidden: int = 128               # 隐层维度
    heads: int = 4                  # 多头注意力头数
    layers: int = 2                 # 异构图注意力层数
    history: int = 16               # 历史观测窗口长度
    top_b: int = 16                 # 每组状态保留的维度数（稀疏门控用）
    gate_threshold: float = 0.5     # 硬门控激活阈值
    candidate_budgets: tuple[int, ...] = (32,)  # 候选智能体预算（v3 固定为 32）
    dropout: float = 0.05           # dropout 比例
    index: str = "exact"            # 检索索引类型：exact（精确 MIPS）或 lsh（局部敏感哈希）
    index_seed: int = 2718          # 检索索引随机种子
    ablation: str = "none"          # 消融开关（none/no_gates/no_history/no_dag/no_edges）
    prediction_mode: str = "residual"  # 预测模式：residual（残差）或 absolute（绝对）

    @property
    def state_dimension_budget(self):
        """每组状态保留的维度数"""
        return self.top_b

    def __post_init__(self):
        # 参数合法性校验
        if tuple(self.candidate_budgets)!=(32,):
            raise ValueError("v3 candidate budget must be exactly (32,)")
        if self.hidden % self.heads or self.top_b > self.hidden:
            raise ValueError("hidden must divide heads and top_b must not exceed hidden")
        if self.index not in ("exact", "lsh"):
            raise ValueError(self.index)


@dataclass
class Snapshot:
    """决策时刻的环境快照，是模型的唯一输入（不含未来信息）。"""
    time: float                             # 快照时间戳
    observed_at: list[float]                # 每条历史观测的时间戳（需按时间升序）
    states: dict[str, list]                 # 各对象组状态 [对象数, 历史长度, 特征维度]
    missing: dict[str, list]                # 各状态缺失掩码，形状同 states
    skills: list[list[str]]                 # 每个智能体拥有的技能
    tools: list[list[str]]                  # 每个智能体可用的工具 API
    knowledge: list[list[str]]              # 每个智能体接入的知识库
    descriptions: list[str]                 # 智能体能力描述文本
    agent_compute: list[int]                # 智能体 -> 所属算力节点 的映射
    capacities: list[list[int]]             # 算力节点容量：GPU 槽位、内存 GiB、显存 GiB
    links: list[list[int]]                  # 物理连边端点：先是智能体，再是算力节点
    bandwidth: list[int]                    # 每条链路带宽（Mbps），每次传输固定预留
    available: list[float]                  # 各智能体的可用时间
    reservations: list[dict] = field(default_factory=list)  # 已占用的资源预留
    revision: str = ""                      # 快照修订号，拓扑变化时更新以失效缓存（？）
    agent_profiles: list[dict] = field(default_factory=list)  # 各智能体的仿真行为参数
    schema: str = SCHEMA_VERSION            # 数据结构版本号

    def validate(self):
        """校验快照各维度一致、且不包含未来观测。"""
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
        """返回能执行该任务的智能体下标：技能、工具、知识库三项都要满足。"""
        return [i for i, skills in enumerate(self.skills)
                if task.skill in skills and set(task.tools) <= set(self.tools[i])
                and set(task.knowledge) <= set(self.knowledge[i])]

    def path(self, source: int, target: int) -> list[int] | None:
        """返回智能体 source 到 target 经过的物理链路下标序列；不可达返回 None。

        路径固定为 source -> 所属算力节点 -> 目标算力节点 -> target。
        """
        if source == target:
            return []
        n = len(self.skills)
        a, b = self.agent_compute[source] + n, self.agent_compute[target] + n
        # 同一修订号内拓扑不变；拓扑更新改变 revision 时缓存失效。
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
    """工作流中的单个子任务。"""
    id: int
    text: str               # 任务提示词
    skill: str              # 所需技能
    tools: list[str]        # 所需工具 API
    knowledge: list[str]    # 所需知识库
    work: float             # 计算工作量
    demand: list[int]       # 资源需求 [GPU 槽位, 内存 GiB, 显存 GiB]


@dataclass
class Workflow:
    """由若干子任务及其依赖构成的有向无环工作流。"""
    id: str
    tasks: list[Task]
    edges: list[list[float]]                # 依赖边：[父任务, 子任务, 传输大小 MB]
    release: float                          # 释放时间
    deadline: float                         # 截止时间（绝对秒）
    min_reliability: float = 0.9            # 最低可靠性要求
    scenario: str = ""                      # 场景（应急救援/智慧家居/智慧工厂）
    difficulty: str = ""                    # 难度（简单/中等/困难）
    template: str = ""                      # 来源模板 ID

    def validate(self):
        """校验 QoS 与拓扑合法性。"""
        if self.deadline <= self.release or not 0 < self.min_reliability <= 1:
            raise ValueError("invalid QoS")
        if [t.id for t in self.tasks] != list(range(len(self.tasks))):
            raise ValueError("task IDs must be consecutive topological indices")
        if any(not 0 <= a < b < len(self.tasks) or mb < 0 for a, b, mb in self.edges):
            raise ValueError("DAG edges must follow topological order")


@dataclass
class Observation:
    """一次真实执行/传输的日志记录（既用于训练标签，也用于评估）。"""
    kind: str               # "node"（任务执行）或 "edge"（任务间传输）
    task: int               # 任务 id（edge 类型时为父任务）
    agent: int              # 执行/发送的智能体
    duration: float         # 耗时
    success: bool           # 是否成功
    started: float          # 开始时间
    finished: float         # 结束时间
    attempt: int            # 尝试次数
    propensity: float       # 日志策略下选择该动作的概率（用于纠偏）
    target_task: int = -1   # 传输的目标任务
    target_agent: int = -1  # 传输的目标智能体
    failure_reason: str = ""  # 失败原因


@dataclass
class Example:
    """一条训练/评估样本：决策快照 + 工作流 + 执行观测 + 溯源信息。"""
    snapshot: Snapshot
    workflow: Workflow
    observations: list[Observation]
    provenance: dict[str, Any]
    schema: str = SCHEMA_VERSION

    def inputs(self):
        """模型可见的输入（只含快照与工作流，不含观测结果）。"""
        return self.snapshot, self.workflow

    def to_dict(self):
        """转为可 JSON 序列化的普通字典（把 numpy 转成 list）。"""
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
        """从字典还原样本，并做严格校验。"""
        if value.get("schema") != SCHEMA_VERSION:
            raise ValueError("legacy teacher labels require explicit conversion; they are not executions")
        w = dict(value["workflow"])
        w["tasks"] = [Task(**t) for t in w["tasks"]]
        result = cls(Snapshot(**value["snapshot"]), Workflow(**w),
                     [Observation(**o) for o in value["observations"]], value["provenance"])
        # 正式数据集用紧凑数组存储状态，避免上百万个 Python 浮点对象占用内存。
        result.snapshot.states={g:np.asarray(x,dtype=np.float32) for g,x in result.snapshot.states.items()}
        result.snapshot.missing={g:np.asarray(x,dtype=bool) for g,x in result.snapshot.missing.items()}
        result.snapshot.validate()
        result.workflow.validate()
        if any(o.started < result.snapshot.time or o.finished < o.started for o in result.observations):
            raise ValueError("execution labels must occur after the decision snapshot")
        return result


@dataclass
class Prediction:
    """模型预测结果：候选智能体、节点/边上的耗时与成功率。"""
    workflow_id: str
    candidates: dict[int, list[int]]                        # 任务 id -> 候选智能体列表
    nodes: dict[tuple[int, int], tuple[float, float]]       # (任务, 智能体) -> (耗时, 成功率)
    edges: dict[tuple[int, int, int, int], tuple[float, float]]  # (父,子,源,目标) -> (耗时, 成功率)
    timing: dict[str, float] = field(default_factory=dict)      # 各阶段耗时统计
    diagnostics: dict[str, Any] = field(default_factory=dict)   # 诊断信息（门控、候选数等）

    def validate(self):
        """校验预测值的数值合法性。"""
        for duration, probability in list(self.nodes.values()) + list(self.edges.values()):
            if not math.isfinite(duration) or duration < 0 or not 0 < probability <= 1:
                raise ValueError("invalid prediction")


@dataclass
class Plan:
    """CP-SAT 求解得到的调度方案。"""
    assignments: list[dict]         # 任务 -> 智能体 的分配
    transfers: list[dict]           # 任务间的传输安排
    admitted: dict[str, bool]       # 各工作流是否被接纳
    status: str                     # 求解器状态
    elapsed: float                  # 求解耗时
    reasons: dict[str, str] = field(default_factory=dict)       # 未接纳原因
    diagnostics: dict[str, Any] = field(default_factory=dict)   # 目标值等诊断信息
