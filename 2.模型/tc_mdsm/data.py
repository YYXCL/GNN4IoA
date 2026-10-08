""""第1步"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .types import Example, Observation, Snapshot, Task, Workflow, SCHEMA_VERSION

# 三个业务场景与三个难度等级
SCENARIOS = ("应急救援", "智慧家居", "智慧工厂")
DIFFICULTIES = ("简单", "中等", "困难")
GENERATOR_VERSION = "3-overlapping-individual-dynamic-first-pass"


def balanced_hosts(agents, computes, rng):
    """生成"智能体 → 算力节点"的均衡映射。"""
    return rng.permutation(np.arange(agents) % computes).astype(int).tolist()


def seed_of(*parts):
    """根据给定的任意参数派生一个确定性的整数种子。"""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def rounded(x):
    """把数组四舍五入到 5 位小数并转为 list。"""
    return np.asarray(x).round(5).tolist()


def read_examples(path):
    """按行读取 gzip 压缩的 jsonl 样本文件，逐条还原为 Example。"""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield Example.from_dict(json.loads(line))


def write_examples(path, examples):
    """把样本逐行写入 gzip 压缩的 jsonl 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")


class Foundation:
    """数据基座：只加载文档化的原始数据源，不读取、不依赖那些由旧教师模型/旧系统/旧规则产出的标签数据。"""

    def __init__(self, root):
        root = Path(root)
        self.root = root
        # 三类原始数据：智能体能力、任务模板、算力负载时序
        agent_path = root / "3.智能体能力数据" / "agent_capabilities.json"
        task_path = root / "5.任务数据/tasks.json"
        load_path = root / "1.算力负载数据/compute_node_load_timeseries.csv"
        agents = json.loads(agent_path.read_text(encoding="utf-8-sig"))
        self.tasks = json.loads(task_path.read_text(encoding="utf-8-sig"))
        self.types = [dict(a, all_skills=[s["id"] for s in a["skills"]]) for a in agents]
        # 汇总"技能 -> 所需工具/知识库"的映射
        self.skill_requirements = {}
        for a in agents:
            for skill in a["skills"]:
                self.skill_requirements.setdefault(skill["id"], {
                    "tools": list(skill.get("required_tools", [])),
                    "knowledge": list(skill.get("required_knowledge", []))})
        # 按时间戳聚合算力负载时序：[GPU剩余(TFLOPS), 内存剩余(GB), 显存剩余(GB), 队列响应(s), 负载率]
        by_time = defaultdict(list)
        with load_path.open(encoding="utf-8-sig", newline="") as handle:
            for r in csv.DictReader(handle):
                by_time[int(r["时间戳"])].append([
                    float(r["GPU算力剩余量(TFLOPS)"]), float(r["内存剩余量(GB)"]),
                    float(r["显存剩余量(GB)"]), float(r["队列响应时间(ms)"]) / 1000,
                    float(r["服务器负载率代理"]),
                ])
        self.timestamps = sorted(by_time)
        self.loads = np.asarray([by_time[t] for t in self.timestamps], dtype=np.float32)
        # 按"场景 x 难度"分层，再按确定性顺序切分 train/val/test
        self.partitions = {s: [] for s in ("train", "val", "test")}
        strata = defaultdict(list)
        for t in self.tasks:
            strata[t["场景"], t["难度级别"]].append(t)
        for items in strata.values():
            items.sort(key=lambda t: seed_of(t["任务id"], "template-split"))
            for index, t in enumerate(items):
                split = "train" if index < len(items) - 2 else "val" if index == len(items) - 2 else "test"
                self.partitions[split].append(t)
        # 记录源文件哈希，保证数据可复现
        self.source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (agent_path, task_path, load_path)}

    def bounds(self, split, history=16):
        """返回某切分可用的时序窗口下标范围，并做 history 长度的隔离（purge）。"""
        n = len(self.timestamps)
        a, b = int(n * .75), int(n * .875)
        lo, hi = {"train": (0, a), "val": (a + history, b), "test": (b + history, n)}[split]
        if hi - lo < history:
            raise ValueError("insufficient trace timestamps after purging")
        return lo, hi

    def episode(self, split, index, seed=7, agents=100, computes=None, history=16,
                scenario=None, difficulty=None, stress="normal", severity=1.0, batch=1):
        """生成一个（或一批）仿真样本：合成快照 + 工作流 + 日志策略产生的观测。"""
        computes = computes or max(2, round(agents / 10))
        rng = np.random.default_rng(seed_of(seed, split, index, agents, stress, severity))
        lo, hi = self.bounds(split, history)
        start = int(rng.integers(lo, hi - history + 1))
        slots = rng.integers(0, self.loads.shape[1], computes)
        loads = self.loads[start:start + history, slots].transpose(1, 0, 2).copy()
        # 容量与 TFLOPS 换算属于显式合成假设，并非真实服务器测量值。
        loads[:, :, 0] = 20 + 180 * loads[:, :, 0] / 4000
        loads[:, :, 1] = 64 * np.clip(1 - loads[:, :, 4], .05, .95)
        loads[:, :, 2] = 32 * np.clip(1 - loads[:, :, 4], .05, .95)
        loads[:, :, 3] = np.clip(loads[:, :, 3], 0, 20)
        if stress in ("compute", "combined"):
            # 算力压力：算力逐时间衰减，队列响应上升
            ramp = np.linspace(0, min(.8, .35 * severity), history)
            loads[:, :, :3] *= (1 - ramp)[None, :, None]
            loads[:, :, 3] += 8 * ramp
        # 后台内存占用对请求不可见；此处保守的准入预算在本回合内保持固定。
        capacities = [[4, int(loads[c, -1, 1]), int(loads[c, -1, 2])] for c in range(computes)]
        agent_compute = balanced_hosts(agents, computes, rng)
        descriptions, skills, tools, knowledge, a_states, r_states = [], [], [], [], [], []
        for i in range(agents):
            source = self.types[i % len(self.types)]
            skills.append(source["all_skills"])
            tools.append(source["工具API权限"].split("|"))
            kb = source["接入知识库"]
            knowledge.append([] if kb == "无" else kb.split("|"))
            descriptions.append(source["智能体能力描述"])
            busy = rng.uniform(0, .4)
            # A 组状态：[上下文窗口, 占用率, 工具速率限制, 忙碌度, 速度, 熟练度]
            a_states.append(np.tile([source["上下文窗口总量(tokens)"], rng.uniform(.1, .7),
                                     source["工具API速率限制(次/分钟)"], busy, source.get("simulation_profile",{}).get("speed",1.), source.get("simulation_profile",{}).get("proficiency",.9)], (history, 1)))
            fail = rng.uniform(.01, .07)
            trend = rng.uniform(-.025, .025) * np.linspace(0, 1, history)
            if stress in ("tool", "combined"):
                # 工具压力：失败率随时间上升
                trend += np.linspace(0, .025 * severity, history)
            failures = np.clip(fail + trend, .0002, .15)
            # R 组状态：[成功率, 平均时延, 失败率, 调用计数]
            r_states.append(np.stack([1 - failures, np.full(history, rng.uniform(2, 10)), failures * .6,
                                      np.arange(history) + rng.integers(20, 200)], axis=-1))
        # 物理连边：智能体到其所属算力节点，以及算力节点之间的全连接
        links = [[i, agents + agent_compute[i]] for i in range(agents)]
        links += [[agents + i, agents + j] for i in range(computes) for j in range(i + 1, computes)]
        e_states, bandwidth = [], []
        for _ in links:
            bw = int(rng.choice([100, 500, 1000, 2000]))
            lat = rng.uniform(.002, .05)
            loss = rng.uniform(.0001, .002)
            drift = np.linspace(0, rng.uniform(-.1, .2), history)
            if stress in ("link", "combined"):
                # 链路压力：时延随时间的漂移加剧
                drift += np.linspace(0, .6 * severity, history)
            # E 组状态：[时延, 带宽, 丢包率, 抖动]
            e_states.append(np.stack([lat * (1 + drift), np.full(history, bw),
                                      np.maximum(0, loss + .008 * drift), lat * .1 * (1 + drift)], axis=-1))
            bandwidth.append(bw)
        states = {"A": rounded(a_states), "C": rounded(loads), "E": rounded(e_states), "R": rounded(r_states)}
        # 构造快照（此刻尚无任何观测缺失，missing 全为 False）
        snapshot = Snapshot(0.0, [float(i - history + 1) for i in range(history)], states,
                            {g: [[False] * history for _ in values] for g, values in states.items()},
                            skills, tools, knowledge, descriptions, agent_compute, capacities, links, bandwidth,
                            [float(loads[c, -1, 3]) for c in agent_compute], revision=f"{split}-{seed}-{index}-{agents}-{stress}-{severity}")
        snapshot.agent_profiles = [dict(self.types[i % len(self.types)].get("simulation_profile",{})) for i in range(agents)]
        # 用同一 World 随机过程回填 R 组中的工具压力与算力相关时延，保证与执行环境一致
        from .world import World
        history_world=World(snapshot,seed_of(seed,split,index,"world"))
        for agent in range(agents):
            speed=snapshot.agent_profiles[agent].get("speed",1.)
            for h,observed in enumerate(snapshot.observed_at):
                tool_pressure=float(np.mean([max(0.,history_world.wave(("tool",tool),observed)) for tool in tools[agent]])) if tools[agent] else 0.
                snapshot.states["R"][agent][h][2]=.10*tool_pressure
                snapshot.states["R"][agent][h][1]=400/max(4,loads[agent_compute[agent],h,0]*speed)

        # 从目标分层里挑选工作流模板
        choices = [t for t in self.partitions[split] if (scenario is None or t["场景"] == scenario)
                   and (difficulty is None or t["难度级别"] == difficulty)]
        if not choices:
            raise ValueError("no source template for requested stratum")
        examples = []
        for b in range(batch):
            source = choices[int(rng.integers(len(choices)))]
            task_rows = source["子任务"]
            tasks = []
            for k, row in enumerate(task_rows):
                skill = row["必需的Skill id"]
                requirement=self.skill_requirements.get(skill,{"tools":[],"knowledge":[]})
                required_tools=requirement["tools"]
                required_kb=requirement["knowledge"]
                tasks.append(Task(k, row["任务具体提示词"], skill, required_tools, required_kb,
                                  float(rng.uniform(150, 800)), [1, int(rng.choice([4, 8, 16])), int(rng.choice([2, 4, 8]))]))
            # 把源 DAG 边映射为任务下标，并赋随机传输大小
            task_ids = {r["任务id"]: i for i, r in enumerate(task_rows)}
            raw_edges = source["DAG"]["边"]
            edges = []
            for e in raw_edges:
                if isinstance(e, dict):
                    parent, child = e.get("source", e.get("from")), e.get("target", e.get("to"))
                else:
                    parent, child = e[:2]
                if parent in task_ids and child in task_ids:
                    edges.append([task_ids[parent], task_ids[child], float(rng.uniform(1, 40))])
            if len(tasks) > 1 and not edges:
                raise ValueError(f"unrecognized source DAG: {raw_edges}")
            deadline = float({"简单": 80, "中等": 160, "困难": 240}[source["难度级别"]])
            workflow = Workflow(f"{split}-{seed}-{index}-{b}", tasks, edges, 0, deadline, .9,
                                source["场景"], source["难度级别"], source["任务id"])
            # 溯源信息：记录生成器版本、切分、种子、压力设置与数据来源
            provenance = {"generator_version": GENERATOR_VERSION, "split": split, "seed": seed, "episode": index, "batch": b,
                          "world_seed": seed_of(seed, split, index, "world"), "stress": stress,
                          "severity": severity, "trace_group": f"{split}:{start // history}",
                          "trace_start": self.timestamps[start], "trace_end": self.timestamps[start + history - 1],
                          "template_group": source["任务id"], "origin": "trace-backed simulation, not measured execution"}
            example = Example(snapshot, workflow, [], provenance)
            # 观测由日志策略（混合探索）生成
            example.observations = logging_policy(example)
            examples.append(example)
        return examples


def logging_policy(example):
    """日志策略：混合探索。只执行因果就绪的任务和被真实选择的传输。"""
    from .world import World
    s, w = example.inputs()
    world = World(s, example.provenance["world_seed"])
    rng = np.random.default_rng(seed_of(example.provenance["world_seed"], w.id, "logging"))
    chosen, finished, logs = {}, {}, []
    for task in w.tasks:
        # 只保留硬可行性（权限 + 资源容量）都满足的智能体
        feasible = [i for i in s.feasible(task)
                    if all(d <= c for d, c in zip(task.demand, s.capacities[s.agent_compute[i]]))]
        if not feasible:
            continue
        # 以 50% 均匀 + 25% 算力最强 + 25% 可靠性最高 的混合概率选择智能体
        speed = np.asarray([s.states["C"][s.agent_compute[i]][-1][0] for i in feasible])
        rel = np.asarray([s.states["R"][i][-1][0] for i in feasible])
        probability = np.full(len(feasible), .5 / len(feasible))
        probability[int(speed.argmax())] += .25
        probability[int(rel.argmax())] += .25
        position = int(rng.choice(len(feasible), p=probability))
        agent = feasible[position]
        ready = max(s.available[agent], max((o.finished for o in logs),default=0))
        # 检查所有前驱任务是否都已完成
        parents = [(int(a), mb) for a, b, mb in w.edges if b == task.id]
        if any(a not in finished for a, _ in parents):
            continue
        valid = True
        for parent, mb in parents:
            path = s.path(chosen[parent], agent)
            if path is None:
                valid = False
                break
            start = max(finished[parent], max((o.finished for o in logs),default=0))
            duration, success = world.transfer(w, parent, task.id, chosen[parent], agent, mb, start, 0)
            logs.append(Observation("edge", parent, chosen[parent], duration, success, start, start + duration,
                                    0, float(probability[position]), task.id, agent, world.last_reason))
            ready = max(ready, start + duration)
            if not success:
                return logs
        if not valid:
            continue
        duration, success = world.execute(w, task.id, agent, ready, 0)
        logs.append(Observation("node", task.id, agent, duration, success, ready, ready + duration,
                                0, float(probability[position]), failure_reason=world.last_reason))
        chosen[task.id] = agent
        if success:
            finished[task.id] = ready + duration
        else:
            return logs
    return logs


def build_dataset(foundation, output, counts=(12000, 1500, 1500), seed=20260917, agents=100):
    """构建完整的 train/val/test 数据集并写出 manifest。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
                "seed": seed, "source_hashes": foundation.source_hashes,
                "assumptions": ["virtual servers, 4 GPU slots / 64 GiB RAM / 32 GiB VRAM",
                                "synthetic service work, permissions requirements, reliability and communication",
                                "balanced random host assignment independent of capability type",
                                "memory admission budgets equal floored remaining RAM/VRAM at snapshot, fixed per episode",
                                "reused trace windows are grouped, not independent real measurements"], "splits": {}}
    for split, count in zip(("train", "val", "test"), counts):
        path = output / f"{split}.jsonl.gz"
        # 每个样本轮换场景/难度，保证各分层均匀覆盖
        examples = (foundation.episode(split, i, seed, agents=agents,
                                      scenario=SCENARIOS[i % 3], difficulty=DIFFICULTIES[(i // 3) % 3],
                                      stress="normal")[0] for i in range(count))
        write_examples(path, examples)
        manifest["splits"][split] = {"count": count, "trace_bounds": foundation.bounds(split),
                                     "templates": [t["任务id"] for t in foundation.partitions[split]],
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
