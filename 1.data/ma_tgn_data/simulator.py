"""智能体—算力节点异构图仿真器。"""

from __future__ import annotations

import random
from typing import Dict, List

import torch

from ma_tgn.config import ModelConfig
from ma_tgn.state_schema import NORMALIZATION_SCALES
from ma_tgn.types import EventType, GraphEvent, QueryGraph, RelationType, TaskRequirement, TrainingCase

Tensor = torch.Tensor

ROLE_CAPABILITIES = {
    "vision": ("多模态视觉分析", ("判断单个视频帧是否有火灾", "判断单个视频帧是否有烟雾", "识别视频帧中的人员")),
    "planning": ("复杂任务规划", ("拆分复杂任务", "生成子任务依赖顺序", "选择协作角色")),
    "retrieval": ("知识检索", ("检索网络运维知识", "检索领域文档证据", "核对检索结果来源")),
    "coding": ("代码生成与执行", ("生成 Python 代码", "修复代码缺陷", "解释程序执行结果")),
    "network": ("6G 网络控制", ("诊断链路拥塞", "规划边云任务迁移", "检测网络异常")),
    "tool": ("工具调用", ("调用外部 API", "调用知识库", "验证工具执行结果")),
}

TASK_QOS = {
    "fire_frame": (0.90, 0.94), "code_fix": (0.55, 0.90),
    "network_diagnosis": (0.95, 0.82), "knowledge_answer": (0.65, 0.94),
    "task_planning": (0.60, 0.90),
}

TASK_CAPABILITY = {
    "fire_frame": "判断单个视频帧是否有火灾",
    "code_fix": "修复代码缺陷",
    "network_diagnosis": "诊断链路拥塞",
    "knowledge_answer": "检索领域文档证据",
    "task_planning": "拆分复杂任务",
}


def _clip(value: Tensor) -> Tensor:
    return value.clamp(0, 1)


class SixGStateSimulator:
    def __init__(self, config: ModelConfig, seed: int = 7, update_interval_ms: int = 100):
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.update_interval_ms = update_interval_ms
        self.roles = [self.rng.choice(tuple(ROLE_CAPABILITIES)) for _ in range(config.num_agents)]
        self.agent_skills = [tuple(ROLE_CAPABILITIES[role][1]) for role in self.roles]
        self.agent_reliability = [torch.tensor([
            self.rng.uniform(0.78, 0.98), self.rng.uniform(500, 12_000),
            self.rng.uniform(0.01, 0.12), self.rng.uniform(0.005, 0.08),
        ]) for _ in self.roles]
        self.agent_states = [self._initial_agent(index) for index in range(config.num_agents)]
        self.compute_kinds = ["cloud" if index < max(1, config.num_compute_nodes // 3) else "edge" for index in range(config.num_compute_nodes)]
        self.compute_states = [self._initial_compute(kind) for kind in self.compute_kinds]
        # 云节点可承载更多智能体，边节点承载较少；按权重建立固定接入关系。
        weights = [5 if kind == "cloud" else 1 for kind in self.compute_kinds]
        population = list(range(config.num_compute_nodes))
        self.agent_compute = [self.rng.choices(population, weights=weights, k=1)[0] for _ in range(config.num_agents)]

    def _initial_agent(self, index: int) -> Dict[str, Tensor]:
        window = float(self.rng.choice((8_192, 32_768, 131_072, 262_144)))
        context = torch.tensor([window, self.rng.uniform(0.05, 0.70)])
        environment = torch.tensor([
            self.rng.uniform(0.35, 1.0), self.rng.uniform(0.0, 0.75),
            self.rng.uniform(0.35, 1.0), self.rng.uniform(0.0, 0.75),
            self.rng.uniform(0.4, 1.0), self.rng.uniform(0.0, 1.0), self.rng.uniform(0.2, 1.0),
        ])
        return {
            "context": context, "workload": torch.tensor([self.rng.uniform(0, 8)]),
            "environment": environment, "reliability": self.agent_reliability[index].clone(),
        }

    def _initial_compute(self, kind: str) -> Tensor:
        if kind == "cloud":
            total_flops = self.rng.uniform(2_000, 9_000); vram = self.rng.uniform(160, 512); bus = self.rng.uniform(2_000_000, 8_000_000)
            utilization = self.rng.uniform(0.18, 0.72); queue = self.rng.randint(0, 24); max_batch = self.rng.choice((32, 64, 128))
        else:
            total_flops = self.rng.uniform(40, 480); vram = self.rng.uniform(16, 96); bus = self.rng.uniform(100_000, 900_000)
            utilization = self.rng.uniform(0.30, 0.92); queue = self.rng.randint(0, 48); max_batch = self.rng.choice((4, 8, 16))
        memory_total = 1024.0 if kind == "cloud" else 256.0
        return torch.tensor([total_flops * (1-utilization), memory_total * (1-utilization), vram * (1-utilization)])

    def _agent_state(self, agent_id: int) -> Dict[str, Tensor]:
        state = self.agent_states[agent_id]
        state["reliability"] = self.agent_reliability[agent_id].clone()
        return {key: value.clone() for key, value in state.items()}

    def _compute_state(self, compute_index: int) -> Dict[str, Tensor]:
        return {"compute": self.compute_states[compute_index].clone()}

    def _step_agent(self, agent_id: int) -> None:
        state = self.agent_states[agent_id]
        state["context"][1] = float(_clip(0.86 * state["context"][1] + 0.14 * torch.tensor(self.rng.uniform(0.02, 1.0))))
        drift = torch.randn(7, generator=self.gen) * 0.04
        state["environment"] = _clip(0.90 * state["environment"] + 0.10 * _clip(state["environment"] + drift))
        state["workload"] = (0.8 * state["workload"] + 0.2 * torch.tensor([self.rng.uniform(0, 10)])).clamp(0, 16)

    def _step_compute(self, compute_index: int) -> None:
        state = self.compute_states[compute_index]
        factors = torch.tensor([self.rng.uniform(0.90, 1.08) for _ in range(3)])
        state.mul_(factors).clamp_(min=0.0)

    def _link(self, src_type: int, dst_type: int) -> Tensor:
        # 路径性质只影响仿真分布，不写入图特征。
        if src_type != dst_type:
            latency = self.rng.uniform(5, 80); bandwidth = self.rng.uniform(100, 20_000)
        else:
            latency = self.rng.uniform(1, 160); bandwidth = self.rng.uniform(50, 50_000)
        loss = min(0.25, max(0.0, self.rng.betavariate(1.2, 35)))
        jitter = max(0.1, latency * self.rng.uniform(0.02, 0.30))
        return torch.tensor([latency, bandwidth, loss, jitter])

    def _task_feedback(self, agent_id: int, capability: str) -> None:
        del capability
        compute = self.compute_states[self.agent_compute[agent_id]]
        remaining = max(1.0, float(compute[1]))
        execution_ms = min(120_000.0, 400.0 + 240_000.0 / remaining + self.rng.uniform(0, 900))
        environment = self.agent_states[agent_id]["environment"]
        success_probability = 0.55 + 0.35 * float(environment.mean())
        success = self.rng.random() < success_probability
        tool_failed = self.rng.random() < float(1-environment[[0, 2]].mean()) * 0.35
        hallucinated = self.rng.random() < (0.06 if success else 0.28)
        old = self.agent_reliability[agent_id]
        observation = torch.tensor([float(success), execution_ms, float(tool_failed), float(hallucinated)])
        self.agent_reliability[agent_id] = 0.9 * old + 0.1 * observation

    def make_case(self, max_candidates: int = 16, event_count: int = 24) -> TrainingCase:
        task_name = self.rng.choice(tuple(TASK_QOS))
        fine_capability = TASK_CAPABILITY[task_name]
        task = TaskRequirement(
            torch.tensor(TASK_QOS[task_name]), torch.empty(0),
            description=task_name, required_capability=fine_capability,
            dag_descriptions=(task_name,), dag_required_capabilities=(fine_capability,),
            dag_qos=torch.tensor([TASK_QOS[task_name]]), dag_edge_index=torch.empty(2, 0, dtype=torch.long),
        )
        candidates = torch.tensor(self.rng.sample(range(self.config.num_agents), min(max_candidates, self.config.num_agents)))
        events: List[GraphEvent] = []
        timestamp = 0.0
        for event_index in range(event_count):
            timestamp += self.update_interval_ms / 1000.0
            # 查询候选必须在历史事件中至少出现一次，否则其 TGN 记忆为零，
            # 排序学习会退化为对未观测节点的随机猜测。
            agent = int(candidates[event_index % candidates.numel()])
            self._step_agent(agent)
            if event_index % 2 == 0:
                compute_index = self.agent_compute[agent]
                self._step_compute(compute_index)
                compute_id = self.config.num_agents + compute_index
                src, dst, src_type, dst_type = compute_id, agent, 1, 0
                src_state, dst_state = self._compute_state(compute_index), self._agent_state(agent)
                relation, event_type = RelationType.COMPUTE_ACCESS, EventType.COMPUTE_UPDATE
            else:
                other = self.rng.choice([i for i in range(self.config.num_agents) if i != agent])
                self._step_agent(other)
                if event_index % 5 == 0:
                    self._task_feedback(agent, fine_capability)
                    event_type = EventType.TASK_RESULT
                else:
                    event_type = self.rng.choice((EventType.CONTEXT_UPDATE, EventType.ENVIRONMENT_UPDATE, EventType.LINK_UPDATE))
                src, dst, src_type, dst_type = other, agent, 0, 0
                src_state, dst_state = self._agent_state(other), self._agent_state(agent)
                relation = RelationType.COMMUNICATION
            confidence = self.rng.uniform(0.82, 0.99)
            observed_delay = self.rng.uniform(0.0, 0.35)
            events.append(GraphEvent(
                src=src, dst=dst, src_node_type=src_type, dst_node_type=dst_type,
                timestamp=timestamp, event_type=event_type, relation_type=relation,
                src_state=src_state, dst_state=dst_state, edge_state=self._link(src_type, dst_type),
                src_confidence={"compute" if src_type else "context": torch.tensor([confidence]), **({} if src_type else {"environment": torch.tensor([confidence])})},
                dst_confidence={"compute" if dst_type else "context": torch.tensor([confidence]), **({} if dst_type else {"environment": torch.tensor([confidence])})},
                src_observed_at={"compute" if src_type else "context": timestamp-observed_delay, **({} if src_type else {"environment": timestamp-observed_delay})},
                dst_observed_at={"compute" if dst_type else "context": timestamp-observed_delay, **({} if dst_type else {"environment": timestamp-observed_delay})},
                edge_confidence=confidence, edge_observed_at=timestamp-observed_delay,
            ))

        sources: list[int] = []; destinations: list[int] = []; edges: list[Tensor] = []
        relations: list[int] = []; edge_times: list[float] = []; edge_confidences: list[float] = []
        candidate_compute: dict[int, int] = {}
        candidate_link: dict[int, Tensor] = {}
        for dst in candidates.tolist():
            compute_index = self.agent_compute[dst]
            compute_id = self.config.num_agents + compute_index
            candidate_compute[dst] = compute_index
            compute_link = self._link(1, 0); candidate_link[dst] = compute_link
            sources.append(compute_id); destinations.append(dst); edges.append(compute_link); relations.append(RelationType.COMPUTE_ACCESS)
            edge_times.append(timestamp-self.rng.uniform(0, .5)); edge_confidences.append(self.rng.uniform(.85, .99))
            peers = self.rng.sample([i for i in range(self.config.num_agents) if i != dst], min(2, self.config.num_agents-1))
            for peer in peers:
                sources.append(peer); destinations.append(dst); edges.append(self._link(0, 0)); relations.append(RelationType.COMMUNICATION)
                edge_times.append(timestamp-self.rng.uniform(0, .8)); edge_confidences.append(self.rng.uniform(.80, .98))
        graph = QueryGraph(
            candidate_ids=candidates, edge_index=torch.tensor([sources, destinations]), edge_state=torch.stack(edges),
            relation_type=torch.tensor(relations), edge_timestamp=torch.tensor(edge_times),
            edge_confidence=torch.tensor(edge_confidences), query_time=timestamp+0.1,
            candidate_skill_ids=tuple(
                self.agent_skills[index] for index in candidates.tolist()
            ),
        )

        teacher_scores: list[Tensor] = []; targets: list[Tensor] = []
        for agent_id in candidates.tolist():
            exact = fine_capability in self.agent_skills[agent_id]
            capability_similarity = float(exact)
            reliability = self.agent_reliability[agent_id]
            state = self.agent_states[agent_id]
            context_score = (1-state["context"][1]) * min(1.0, float(state["context"][0])/32_768)
            environment_score = state["environment"].mean()
            compute = self.compute_states[candidate_compute[agent_id]]
            compute_score = (compute / torch.tensor([9000.0, 1024.0, 512.0])).clamp(0, 1).mean()
            link = candidate_link[agent_id]
            link_score = (1-min(1, link[0]/1000)) * min(1, link[1]/10_000) * (1-link[2]) * (1-min(1, link[3]/200))
            execution_norm = min(1.0, float(reliability[1])/120_000)
            executability = float(environment_score * min(1.0, capability_similarity))
            score = (
                0.24*capability_similarity + 0.12*context_score + 0.16*environment_score + 0.20*compute_score + 0.10*link_score
                + 0.09*float(reliability[0]) + 0.04*(1-execution_norm) + 0.025*(1-float(reliability[2])) + 0.025*(1-float(reliability[3]))
            )
            teacher_scores.append(torch.as_tensor(score))
            targets.append(torch.tensor([
                float(reliability[0]), execution_norm, float(reliability[2]),
                float(reliability[3]), executability,
            ]))
        scores = torch.stack(teacher_scores)
        return TrainingCase(
            events, graph, task, scores.argmax(), torch.stack(targets), scores,
            {
                "task_type": task_name, "required_capability": fine_capability,
                "update_interval_ms": self.update_interval_ms,
                "candidate_roles": [self.roles[int(i)] for i in candidates],
                "agent_roles": self.roles, "agent_compute": self.agent_compute,
                "compute_kinds": self.compute_kinds,
                "compute_states": [state.tolist() for state in self.compute_states],
            },
        )
