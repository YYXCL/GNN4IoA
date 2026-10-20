"""实验配置及快照、工作流、观测、预测和调度方案。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections import deque

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
    candidate_budgets: tuple[int, ...] = (8,)  # 在线候选预算
    planning_batch_size: int = 4    # 每次联合规划的最大工作流数
    dropout: float = 0.05           # dropout 比例
    ablation: str = "none"          # 消融开关（none/no_gates/no_history/no_dag/no_edges）
    prediction_mode: str = "residual"  # 预测模式：residual（残差）或 absolute（绝对）
    deadline_policy: str = "hard"   # hard:预测截止硬约束；evaluate:实际执行后判断迟到
    solver_start: str = "serial"    # serial:可行串行初始提示；reject:原全部拒绝提示，用于对照
    execution_policy: str = "ready" # ready:实际就绪启动；planned:计划时刻作为启动下界

    @property
    def state_dimension_budget(self):
        """每组状态保留的维度数"""
        return self.top_b

    def __post_init__(self):
        # 参数合法性校验
        if len(self.candidate_budgets)!=1 or self.candidate_budgets[0]<=0 or self.planning_batch_size<=0:
            raise ValueError("candidate budget and planning batch size must be positive")
        if self.hidden % self.heads or self.top_b > self.hidden:
            raise ValueError("hidden must divide heads and top_b must not exceed hidden")
        if self.deadline_policy not in ("hard","evaluate"):
            raise ValueError("deadline_policy must be hard or evaluate")
        if self.solver_start not in ("serial","reject"):
            raise ValueError("solver_start must be serial or reject")
        if self.execution_policy not in ("ready","planned"):
            raise ValueError("execution_policy must be ready or planned")


@dataclass
class Snapshot:
    """决策时刻的环境快照，是模型的唯一输入（不含未来信息）。"""
    time: float                        # 快照时间戳
    observed_at: list[float]           # 每条历史观测的时间戳（需按时间升序）
    states: dict[str, list]                 # 各对象组状态 [对象数, 历史长度, 特征维度]
    missing: dict[str, list]                # 整步缺失掩码 [对象数, 历史长度]
    skills: list[list[str]]                 # 每个智能体拥有的技能
    tools: list[list[str]]                  # 每个智能体可用的工具 API
    knowledge: list[list[str]]              # 每个智能体接入的知识库
    descriptions: list[str]                 # 智能体能力描述文本
    agent_compute: list[int]                # 智能体 -> 所属算力节点 的映射
    capacities: list[list[int]]             # 算力节点容量：GPU 槽位、内存 GiB、显存 GiB
    links: list[list[int]]                  # 物理连边端点：先是智能体，再是算力节点
    bandwidth: list[int]                    # 每条链路带宽（Mbps），每次传输固定预留
    available: list[float]             # 各智能体的可用时间
    revision: str = ""                      # 快照修订号，用于状态与路径缓存
    agent_profiles: list[dict] = field(default_factory=list)  # 各智能体的仿真行为参数


    def feasible(self, task: Task) -> list[int]:
        """返回能执行该任务的智能体下标：技能、工具、知识库三项都要满足。"""
        return [i for i, skills in enumerate(self.skills)
                if task.skill in skills and set(task.tools) <= set(self.tools[i])
                and set(task.knowledge) <= set(self.knowledge[i])
                and all(d<=c for d,c in zip(task.demand,self.capacities[self.agent_compute[i]]))]

    def path(self, source: int, target: int) -> list[int] | None:
        """返回智能体 source 到 target 经过的物理链路下标序列；不可达返回 None。

        优先使用直连；无直连时按最少跳数经过其他智能体中转。
        端点也可为算力节点（编号从智能体数起）；算力节点不能作为中转。
        """
        if source == target:
            return []
        n = len(self.skills)
        # 同一修订号内拓扑不变；拓扑更新改变 revision 时缓存失效。
        cache_key=(id(self.links),len(self.links),self.revision)
        if getattr(self,"_path_cache_key",None)!=cache_key:
            self._path_lookup={tuple(sorted(edge)): i for i, edge in enumerate(self.links)}
            self._path_adjacency={}
            for index,(left,right) in enumerate(self.links):
                self._path_adjacency.setdefault(left,[]).append((right,index))
                self._path_adjacency.setdefault(right,[]).append((left,index))
            self._paths={}
            self._path_cache_key=cache_key
        lookup=self._path_lookup
        direct=lookup.get(tuple(sorted((source,target))))
        if direct is not None:return [direct]
        pair=(source,target)
        if pair in self._paths:return self._paths[pair]
        queue=deque([source]);previous={source:None}
        while queue and target not in previous:
            vertex=queue.popleft()
            for neighbor,edge in self._path_adjacency.get(vertex,[]):
                if neighbor in previous or (neighbor>=n and neighbor!=target):continue
                previous[neighbor]=(vertex,edge);queue.append(neighbor)
        result=None
        if target in previous:
            result=[];vertex=target
            while vertex!=source:
                vertex,edge=previous[vertex];result.append(edge)
            result.reverse()
        self._paths[pair]=result
        self._paths[target,source]=None if result is None else list(reversed(result))
        return result


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
    propensity: float       # 观测策略选择概率（当前损失未使用加权纠偏）
    target_task: int = -1   # 传输的目标任务
    target_agent: int = -1  # 传输的目标智能体
    failure_reason: str = ""  # 失败原因


@dataclass
class Example:
    """一条训练/评估样本：决策快照 + 工作流 + 执行观测 + 样本参数。"""
    snapshot: Snapshot
    workflow: Workflow
    observations: list[Observation]
    provenance: dict[str, Any]

    def inputs(self):
        """模型可见的输入（只含快照与工作流，不含观测结果）。"""
        return self.snapshot, self.workflow


@dataclass
class Prediction:
    """模型预测结果：候选智能体、节点/边上的耗时与成功率。"""
    workflow_id: str
    candidates: dict[int, list[int]]                        # 任务 id -> 候选智能体列表
    nodes: dict[tuple[int, int], tuple[float, float]]       # (任务, 智能体) -> (耗时, 成功率)
    edges: dict[tuple[int, int, int, int], tuple[float, float]]  # (父,子,源,目标) -> (耗时, 成功率)


@dataclass
class Plan:
    """CP-SAT 求解得到的调度方案。"""
    assignments: list[dict]         # 任务 -> 智能体 的分配
    transfers: list[dict]           # 任务间的传输安排
    admitted: dict[str, bool]       # 各工作流是否被接纳
    status: str                     # 求解器状态
    elapsed: float                  # 求解耗时
    reasons: dict[str, str] = field(default_factory=dict)       # 未接纳原因
    timings: dict[str, float] = field(default_factory=dict)    # 全局准备/在线预测/建模/求解耗时
