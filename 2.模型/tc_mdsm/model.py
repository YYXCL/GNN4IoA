"""第2步：TC-MDSM 与五个基线方法，共用同一套预测与调度接口。"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .types import GROUP_DIMS, METHODS, ModelConfig, Prediction
from .layers import HeterogeneousLayer
from .retrieval import CandidateIndex


def hash_text(text, dimensions=128):
    """把任务文本哈希成固定维度的稀疏特征向量（1/2/3-gram 的符号散列）。"""
    vector = np.zeros(dimensions, np.float32)
    text = text.lower().replace(" ", "")
    for n in (1, 2, 3):
        for p in range(max(0, len(text) - n + 1)):
            key = int.from_bytes(hashlib.blake2b(text[p:p+n].encode(), digest_size=8).digest(), "little")
            vector[key % dimensions] += 1 if key & 256 else -1
    return vector / max(1, float(np.linalg.norm(vector)))


class Predictor(nn.Module):
    """核心预测模型：编码状态与任务，检索候选智能体，在异构图上传导并预测耗时/成功率。"""

    def __init__(self, config=None, method="TC-MDSM"):
        super().__init__()
        self.config = config or ModelConfig()
        self.method = method
        if method not in METHODS:
            raise ValueError(method)
        d, cfg = self.config.hidden, self.config
        # 每组状态的线性编码器（输入维度 = 特征数 + 1 个缺失标志位）
        self.encoders = nn.ModuleDict({g: nn.Linear(n + 1, d) for g, n in GROUP_DIMS.items()})
        # 每组状态的 GRU（用于建模时间序列）
        self.grus = nn.ModuleDict({g: nn.GRU(d, d, batch_first=True) for g in GROUP_DIMS})
        for g, n in GROUP_DIMS.items():
            self.register_buffer(f"mean_{g}", torch.zeros(n))
            self.register_buffer(f"std_{g}", torch.ones(n))
        self.register_buffer("normalizers_fitted", torch.tensor(False))
        # 任务编码器（输入 128 维文本哈希 + 8 维数值特征 = 136）
        self.task = nn.Sequential(nn.Linear(136, d), nn.GELU(), nn.LayerNorm(d))
        # DAG 编码器：融合自身、前驱/后继均值与依赖规模
        self.dag = nn.Sequential(nn.Linear(d * 3 + 2, d), nn.GELU(), nn.LayerNorm(d))
        # 门控：group_gate 决定用哪几组状态，dim_gate 决定每组保留哪些维度
        self.group_gate = nn.Linear(d, 4)
        self.dim_gate = nn.Linear(d, 4 * d)
        self.key = nn.ModuleDict({g: nn.Linear(d, d) for g in GROUP_DIMS})
        self.query = nn.ModuleDict({g: nn.Linear(d, d) for g in GROUP_DIMS})
        # 任务-智能体配对编码器
        self.pair = nn.Sequential(nn.Linear(d * 4, d), nn.GELU(), nn.LayerNorm(d))
        self.types = nn.Embedding(6, d)
        self.edge_input = nn.Sequential(nn.Linear(d * 2 + 5, d), nn.GELU())
        # 异构图注意力层（HM-MATAS 额外开启边更新）
        self.layers = nn.ModuleList(HeterogeneousLayer(d, cfg.heads, edge_updates=method == "HM-MATAS") for _ in range(cfg.layers))
        self.node_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.edge_head = nn.Sequential(nn.Linear(3 * d + 5, d), nn.GELU(), nn.Linear(d, 2))
        # GMATANN 专用的条件序列 VAE 与局部/全局注意力
        self.cvae = nn.LSTM(d * 2, d, batch_first=True)
        self.posterior = nn.Linear(d, 2 * d)
        self.decoder = nn.GRU(d * 2, d, batch_first=True)
        self.local_attention = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.global_attention = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.fusion = nn.Linear(d * 2, d)
        self.attention_scale = nn.Parameter(torch.ones(2))
        # 概率校准参数（[斜率, 截距]，节点/边各一组）
        self.register_buffer("calibration", torch.tensor([[1., 0.], [1., 0.]]))
        self._state_cache = None
        self._index_cache = None
        self._text_cache = {}
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        """Xavier 初始化线性层权重，偏置置零。"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def device(self):
        return self.calibration.device

    @property
    def sparse(self):
        """TC-MDSM和仅去掉历史编码的StaticGraph均使用门控。"""
        return self.method in ("TC-MDSM", "StaticGraph") and self.config.ablation != "no_gates"

    def fit_normalizers(self, examples):
        """仅用训练集统计各组状态的均值/标准差，用于标准化。"""
        accum = {g: [np.zeros(n), np.zeros(n), 0] for g, n in GROUP_DIMS.items()}
        for e in examples:
            if e.provenance["split"] != "train":
                raise ValueError("normalizers may only fit training data")
            for g in GROUP_DIMS:
                x = np.asarray(e.snapshot.states[g], dtype=np.float64)
                mask = ~np.asarray(e.snapshot.missing[g], dtype=bool)
                x = x[mask]
                accum[g][0] += x.sum(0)
                accum[g][1] += (x*x).sum(0)
                accum[g][2] += len(x)
        for g, (total, squared, n) in accum.items():
            mean = total / max(1, n)
            std = np.maximum(squared / max(1, n) - mean*mean, 1e-4)**.5
            getattr(self, f"mean_{g}").copy_(torch.tensor(mean, device=self.device))
            getattr(self, f"std_{g}").copy_(torch.tensor(std, device=self.device))
        self.normalizers_fitted.fill_(True)
        self._state_cache = None
        self._index_cache = None

    def encode_state(self, snapshot):
        """编码快照中的四组状态，返回 (每组当前状态向量 z, 每组完整序列)。"""
        if not bool(self.normalizers_fitted):
            raise RuntimeError("fit training-only normalization before inference")
        cache_key = (id(snapshot), snapshot.revision, tuple(p._version for p in self.parameters()))
        if not self.training and self._state_cache is not None and self._state_cache[0] == cache_key:
            return self._state_cache[1]
        z, sequences = {}, {}
        # 是否使用时序建模（TC-MDSM/Full-State/GMATANN 且未做 no_history 消融）
        temporal = self.method in ("TC-MDSM", "Full-State", "GMATANN") and self.config.ablation != "no_history"
        for g in GROUP_DIMS:
            x = torch.tensor(snapshot.states[g], dtype=torch.float32, device=self.device)
            missing = torch.tensor(snapshot.missing[g], dtype=torch.bool, device=self.device)
            x = (x - getattr(self, f"mean_{g}")) / getattr(self, f"std_{g}")
            x = torch.where(missing[..., None], torch.zeros_like(x), x)   # 缺失值置零
            x = torch.cat([x, missing[..., None].float()], -1)            # 拼接缺失标志位
            sequence = F.gelu(self.encoders[g](x))
            sequences[g] = sequence
            # 时序模式取 GRU 最后隐状态，否则直接取最后一步编码
            z[g] = self.grus[g](sequence)[1][-1] if temporal else sequence[:, -1]
        if not self.training:
            self._state_cache = (cache_key, (z, sequences))
        return z, sequences

    def prepare_snapshot(self, snapshot):
        """状态更新时提前编码并建立检索键；在线请求复用这份快照。"""
        self.eval()
        if self.method == "Rule-Latency":
            return
        with torch.no_grad():
            z, _ = self.encode_state(snapshot)
            cache_key = self._state_cache[0]
            if self._index_cache is None or self._index_cache[0] != cache_key:
                mapping = torch.tensor(snapshot.agent_compute, device=self.device)
                keys = {g: self.key[g](z[g][mapping] if g == "C" else z[g]) for g in ("A", "C", "R")}
                self._index_cache = (cache_key, {g: CandidateIndex(v) for g, v in keys.items()})
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def encode_tasks(self, workflow):
        """编码工作流中的每个任务，并用 DAG 结构信息（前驱/后继/规模）增强。"""
        vectors = []
        for t in workflow.tasks:
            text = "|".join([t.text, t.skill, *t.tools, *t.knowledge])
            if text not in self._text_cache:
                self._text_cache[text] = hash_text(text)
            # 归一化后的数值特征：工作量、资源需求、时限、可靠性、工具/知识数量
            values = [t.work / 800, *[v / n for v, n in zip(t.demand, (4, 64, 32))],
                      (workflow.deadline - workflow.release) / 240, workflow.min_reliability,
                      len(t.tools) / 5, len(t.knowledge) / 5]
            vectors.append(np.concatenate([self._text_cache[text], values]))
        base = self.task(torch.tensor(np.asarray(vectors), dtype=torch.float32, device=self.device))
        pre, suc = torch.zeros_like(base), torch.zeros_like(base)
        counts = base.new_zeros((len(base), 2))
        sizes = base.new_zeros((len(base), 2))
        if self.config.ablation != "no_dag":
            # 汇总前驱/后继的特征均值与依赖传输规模
            for a, b, mb in workflow.edges:
                a, b = int(a), int(b)
                pre[b] = pre[b] + base[a]
                suc[a] = suc[a] + base[b]
                counts[b, 0] += 1
                counts[a, 1] += 1
                sizes[b, 0] += mb / 40
                sizes[a, 1] += mb / 40
        return self.dag(torch.cat([base, pre / counts[:, :1].clamp_min(1), suc / counts[:, 1:].clamp_min(1), sizes], -1))

    def gates(self, q):
        """计算组门控 a 与维度掩码 v；训练时用软门控，推理时用硬门控。"""
        d = self.config.hidden
        a = self.group_gate(q).sigmoid()
        v = self.dim_gate(q).reshape(len(q), 4, d).sigmoid()
        penalty = a.mean() + v.mean()   # 稀疏惩罚项
        if not self.sparse:
            return torch.ones_like(a), torch.ones_like(v), penalty * 0
        if not self.training:
            # 推理阶段：组门控二值化，并保证 A/C/R 至少有一组被激活
            active = (a >= self.config.gate_threshold).to(a.dtype)
            for k in range(len(q)):
                if active[k, [0, 1, 3]].sum() == 0:
                    group = [0, 1, 3][a[k, [0, 1, 3]].argmax().item()]
                    active[k, group] = 1
            # 维度掩码：每组只保留得分最高的 top_b 个维度
            mask = torch.zeros_like(v).scatter_(-1, v.topk(self.config.top_b, -1).indices, 1)
            return active, mask, penalty
        return a, v, penalty

    def forward(self, snapshot, workflow, budget=None, fixed_candidates=None, forced_pairs=None):
        # 一、编码状态与任务，并计算稀疏门控
        z, sequence = self.encode_state(snapshot)
        q = self.encode_tasks(workflow)
        a, masks, sparse_loss = self.gates(q)
        mapping = torch.tensor(snapshot.agent_compute, device=self.device)
        # 二、构造检索用的键/查询（C 组按键为算力节点，需按映射取对应节点状态）
        index_key=(id(snapshot),snapshot.revision,tuple(p._version for p in self.parameters()))
        cached = not self.training and self._index_cache is not None and self._index_cache[0] == index_key
        keys = ({g: self._index_cache[1][g].keys for g in ("A", "C", "R")} if cached else
                {g: self.key[g](z[g][mapping] if g == "C" else z[g]) for g in ("A", "C", "R")})
        queries = {g: self.query[g](q) * masks[:, gi] for g, gi in (("A", 0), ("C", 1), ("R", 3))}
        # retrieval_scores 仅在训练时计算，用于检索的对比学习损失
        retrieval_scores = sum(a[:, gi, None] * (queries[g] @ keys[g].T) / math.sqrt(self.config.top_b)
                               for g, gi in (("A", 0), ("C", 1), ("R", 3))) if self.training else None
        budget = budget or self.config.candidate_budgets[0]
        candidates = {}
        if cached:
            indexes=self._index_cache[1]
        else:
            # 为 A/C/R 三组各建一个候选索引
            indexes = {g: CandidateIndex(keys[g])
                       for gi, g in enumerate(("A", "C", "R"))}
            if not self.training:
                self._index_cache=(index_key,indexes)
        # 三、为每个任务检索候选智能体（并叠加前驱候选作为锚点、按路径时延重排）
        for k, task in enumerate(workflow.tasks):
            feasible = snapshot.feasible(task)   # 硬可行性（权限）过滤
            dimensions = {g: torch.nonzero(masks[k, gi], as_tuple=True)[0]
                          if not self.training and self.sparse else None
                          for g, gi in (("A",0),("C",1),("R",3))}
            if fixed_candidates is not None:
                selected = [i for i in fixed_candidates[k] if i in feasible][:budget]
            else:
                selected_set = set()
                for g, gi in (("A", 0), ("C", 1), ("R", 3)):
                    if self.training or a[k, gi] > 0:
                        selected_set.update(indexes[g].search(queries[g][k], feasible, budget, dimensions[g]))
                # 加入前驱任务已选候选的交集作为锚点，保证依赖可行
                predecessors = [int(p) for p, child, _ in workflow.edges if child == k]
                anchors = {i for p in predecessors for i in candidates[p]} & set(feasible)
                selected_set.update(anchors)
                ordered = sorted(selected_set)
                if ordered:
                    scores = sum(a[k, gi] * indexes[g].scores(queries[g][k], ordered, dimensions[g]) / math.sqrt(self.config.top_b)
                                 for g, gi in (("A",0),("C",1),("R",3))
                                 if self.training or a[k, gi] > 0).detach()
                    # 用所有前驱候选路径的平均时延对候选重排（不使用任何特权标签）。
                    for pos, i in enumerate(ordered):
                        costs = []
                        for p in predecessors:
                            for j in candidates[p]:
                                path = snapshot.path(j, i)
                                if path is not None:
                                    costs.append(sum(snapshot.states["E"][e][-1][0] for e in path))
                        if costs:
                            scores[pos] -= float(np.mean(costs))
                    selected = [ordered[p] for p in torch.argsort(scores, descending=True, stable=True)[:budget].tolist()]
                else:
                    selected = []
            if forced_pairs is not None:
                # 强制把已观测动作对应的候选并进来，保证训练标签一定落在图里
                selected = sorted(set(selected) | {i for kk, i in forced_pairs if kk == k and i in feasible})
            candidates[k] = selected
        pairs = [(k, i) for k in candidates for i in candidates[k]]   # 所有(任务,智能体)候选对
        if not pairs:
            return {"pairs": [], "edge_keys": [], "node": q.new_zeros((0, 2)), "edge": q.new_zeros((0, 2)),
                    "candidates": candidates, "retrieval_scores": retrieval_scores,
                    "sparse_loss": sparse_loss, "aux_loss": q.sum() * 0}
        k_ids = torch.tensor([k for k, _ in pairs], device=self.device)
        i_ids = torch.tensor([i for _, i in pairs], device=self.device)
        c_ids = mapping[i_ids]   # 候选智能体所属的算力节点
        # 四、编码候选对：任务查询与 A/C/R 三组被选中的键拼接后送入配对编码器
        selected_z = [a[k_ids, gi, None] * masks[k_ids, gi] * keys[g][i_ids]
                      for g, gi in (("A", 0), ("C", 1), ("R", 3))]
        h_pair = self.pair(torch.cat([q[k_ids], *selected_z], -1))
        aux_loss = q.sum() * 0
        if self.method == "GMATANN":
            # GMATANN 分支：条件序列 VAE，对序列做重构并为配对特征注入隐变量
            seq = (sequence["A"][i_ids] + sequence["C"][c_ids] + sequence["R"][i_ids]) / 3
            cond = q[k_ids, None].expand(-1, seq.shape[1], -1)
            encoded = self.cvae(torch.cat([seq, cond], -1))[1][0][-1]
            mu, logvar = self.posterior(encoded).chunk(2, -1)
            logvar = logvar.clamp(-8, 5)
            latent = mu + torch.exp(.5 * logvar) * torch.randn_like(mu) if self.training else mu
            reconstruction = self.decoder(torch.cat([latent[:, None].expand_as(seq), cond], -1))[0]
            aux_loss = F.mse_loss(reconstruction, seq.detach()) + .01 * (-.5 * (1 + logvar - mu.square() - logvar.exp()).mean())
            h_pair = h_pair + latent
        # 五、构建整张异构图（每个 DAG 只构建一次）：
        # 节点 = 任务 + 候选智能体 + 算力节点 + 候选对；边 = 多类关系
        agent_ids, compute_ids = sorted(set(i for _, i in pairs)), sorted(set(c_ids.tolist()))
        nt = len(q)
        ai = {v: nt+j for j, v in enumerate(agent_ids)}
        ci = {v: nt+len(ai)+j for j, v in enumerate(compute_ids)}
        offset = nt + len(ai) + len(ci)
        pi = {pair: offset+j for j, pair in enumerate(pairs)}
        x = torch.cat([q, self.types.weight[1].expand(len(ai), -1), self.types.weight[2].expand(len(ci), -1), h_pair])
        kinds = [0]*nt + [1]*len(ai) + [2]*len(ci) + [3]*len(pairs)
        source, target, relations, features, edge_keys, edge_positions, raw_features = [], [], [], [], [], [], []
        path_ids=[]
        zero = q.new_zeros(self.config.hidden)
        def add(u, v, rel, feat):
            source.append(u); target.append(v); relations.append(rel); features.append(feat)
        for k, i in pairs:
            # 任务 <-> 候选对，以及 智能体 <-> 候选对 的双向边
            for u, v, rel in ((k, pi[k,i], 0), (ai[i], pi[k,i], 2)):
                add(u, v, rel, zero); add(v, u, rel+1, zero)
        for i in agent_ids:
            # 智能体 <-> 所属算力节点
            add(ai[i], ci[snapshot.agent_compute[i]], 4, zero)
            add(ci[snapshot.agent_compute[i]], ai[i], 5, zero)
        for parent, child, mb in workflow.edges:
            # 依赖边：连接父/子任务的候选对，并记录物理链路特征
            parent, child = int(parent), int(child)
            for i in candidates[parent]:
                for j in candidates[child]:
                    path = snapshot.path(i, j)
                    if path is None:
                        continue
                    raw = np.asarray([snapshot.states["E"][p][-1] for p in path])
                    # 物理特征：[传输量, 总时延, 带宽下限, 失败概率, 抖动]
                    physical = [mb/40, float(raw[:,0].sum()) if path else 0,
                                float(raw[:,1].min()/2000) if path else 1,
                                float(1-np.prod(1-raw[:,2])) if path else 0,
                                float(raw[:,3].sum()) if path else 0]
                    edge_positions.append(len(features))
                    add(pi[parent,i], pi[child,j], 6, zero)
                    add(pi[child,j], pi[parent,i], 7, zero)
                    edge_keys.append((parent, child, i, j)); raw_features.append(physical);path_ids.append(path)
        if self.method == "HM-MATAS":
            # HM-MATAS：追加全局状态节点与任务顺序节点，补充分配图之外的信息
            global_id = len(x)
            order_offset = global_id + 1
            x = torch.cat([x, q.mean(0, keepdim=True), q])
            kinds += [4] + [5]*nt
            for k in range(nt):
                add(k, global_id, 8, zero); add(global_id, k, 9, zero)
                add(k, order_offset+k, 8, zero); add(order_offset+k, k, 9, zero)
        source_t, target_t, rel_t, kind_t = [torch.tensor(v, dtype=torch.long, device=self.device) for v in (source,target,relations,kinds)]
        ef = torch.stack(features)
        raw_tensor_features=q.new_tensor(raw_features).reshape(-1,5)
        if edge_keys:
            # 把依赖边上路径的 E 组状态聚合后编码为边特征
            longest=max(1,max(map(len,path_ids)))
            padded=torch.tensor([p+[-1]*(longest-len(p)) for p in path_ids],device=self.device)
            valid=padded>=0
            ze=(z["E"][padded.clamp_min(0)]*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
            parents=torch.tensor([p for p,c,i,j in edge_keys],device=self.device)
            children=torch.tensor([c for p,c,i,j in edge_keys],device=self.device)
            encoded=self.edge_input(torch.cat([a[parents,2,None]*masks[parents,2]*ze,
                                              a[children,2,None]*masks[children,2]*ze,raw_tensor_features],-1))
            if self.config.ablation=="no_edges":
                encoded,raw_tensor_features=torch.zeros_like(encoded),torch.zeros_like(raw_tensor_features)
            positions=torch.tensor(edge_positions,device=self.device)
            ef[positions]=encoded;ef[positions+1]=encoded
        # 六、图传导：默认用异构图注意力；GMATANN 改用局部/全局注意力融合
        if self.method != "GMATANN":
            initial_x, initial_edge = x, ef
            for layer in self.layers:
                x, ef = layer(x, kind_t, source_t, target_t, rel_t, ef, initial_x, initial_edge)
        elif self.method == "GMATANN":
            allowed = torch.eye(len(x), dtype=torch.bool, device=self.device)
            allowed[source_t, target_t] = True
            local = self.local_attention(x[None], x[None], x[None], attn_mask=~allowed, need_weights=False)[0][0]
            global_x = self.global_attention(x[None], x[None], x[None], need_weights=False)[0][0]
            x = self.fusion(torch.cat([x+self.attention_scale[0]*local, x+self.attention_scale[1]*global_x], -1))
        # 七、输出头：候选对节点 -> (log 耗时, 成功率 logit)；依赖边同理
        node = self.node_head(x[offset:offset+len(pairs)])
        if edge_keys:
            left = torch.tensor([pi[a,i] for a,b,i,j in edge_keys], device=self.device)
            right = torch.tensor([pi[b,j] for a,b,i,j in edge_keys], device=self.device)
            edge = self.edge_head(torch.cat([x[left], x[right], ef[edge_positions], raw_tensor_features], -1))
        else:
            edge = q.new_zeros((0,2))
        return {"pairs": pairs, "edge_keys": edge_keys, "node": node, "edge": edge,
                "candidates": candidates, "retrieval_scores": retrieval_scores,
                "sparse_loss": sparse_loss, "aux_loss": aux_loss}

    def predict(self, snapshot, workflow, **kwargs):
        """对外预测接口：返回 Prediction（候选、节点/边耗时与成功率）。"""
        if self.method == "Rule-Latency":
            return rule_prediction(snapshot, workflow, **kwargs)
        self.eval()
        with torch.no_grad():
            output = self(snapshot, workflow, **kwargs)
            decoded=self.decode(output,snapshot,workflow,calibrated=True)
            result = []
            # 分别整理节点预测与边预测为 {键: (耗时, 成功率)}
            for kind, keys in enumerate((output["pairs"], output["edge_keys"])):
                duration,probability,_=decoded["node" if kind==0 else "edge"]
                values=torch.stack([duration,probability],dim=-1).detach().cpu().tolist()
                result.append({tuple(k):tuple(v) for k,v in zip(keys,values)})
            prediction = Prediction(workflow.id, output["candidates"], *result)
            return prediction

    def decode(self,output,snapshot,workflow,calibrated=False):
        """把网络原始输出解码为物理量：先验 + 学习到的修正；同机传递是确定性的。"""
        result={}
        for kind,name in enumerate(("node","edge")):
            raw=output[name]
            priors=[]
            if name=="node":
                # 节点先验：耗时 = 工作量 / 算力；成功率 = 可靠性
                for k,i in output["pairs"]:
                    c=snapshot.agent_compute[i]
                    priors.append((workflow.tasks[k].work/max(1,float(snapshot.states["C"][c][-1][0])),
                                   float(snapshot.states["R"][i][-1][0])))
            else:
                # 边先验：传输耗时 = 传输量/带宽 + 时延；成功率 = 链路可靠性连乘
                sizes={(int(a),int(b)):mb for a,b,mb in workflow.edges}
                for a,b,i,j in output["edge_keys"]:
                    path=snapshot.path(i,j)
                    if not path:
                        priors.append((0.,1.))
                    else:
                        physical=np.asarray([snapshot.states["E"][l][-1] for l in path])
                        priors.append((float(8*sizes[a,b]/physical[:,1].min()+physical[:,0].sum()),float(np.prod(1-physical[:,2]))))
            prior=raw.new_tensor(priors).reshape(-1,2)
            if self.config.prediction_mode=="residual":
                # 残差模式：在先验基础上做乘性/加性修正
                duration=prior[:,0]*torch.exp(2*torch.tanh(raw[:,0]))
                logits=torch.logit(prior[:,1].clamp(1e-5,1-1e-5))+raw[:,1]
            else:
                # 绝对模式：直接由网络输出映射
                duration=F.softplus(raw[:,0])*(10 if kind==0 else 1)+(.001 if kind==0 else 0)
                logits=raw[:,1]
            calibrated_logits=logits*self.calibration[kind,0]+self.calibration[kind,1] if calibrated else logits
            probability=calibrated_logits.sigmoid().clamp(1e-6,1-1e-6)
            if name=="edge":
                # 同机（源=目标）传递：耗时 0、成功率 1
                internal=torch.tensor([i==j for a,b,i,j in output["edge_keys"]],device=self.device,dtype=torch.bool)
                duration=torch.where(internal,torch.zeros_like(duration),duration)
                probability=torch.where(internal,torch.ones_like(probability),probability)
            else:
                duration=duration.clamp_min(.001)
            result[name]=(duration,probability,logits)
        return result


def rule_prediction(snapshot, workflow, budget=32, fixed_candidates=None, **kwargs):
    """Rule-Latency 基线：不使用神经网络，直接用可用时间+算力+时延排序选候选。"""
    candidates, nodes, edges = {}, {}, {}
    for t in workflow.tasks:
        feasible = snapshot.feasible(t)
        scores = {i: snapshot.available[i] + t.work/max(1, snapshot.states["C"][snapshot.agent_compute[i]][-1][0])
                  + snapshot.states["E"][i][-1][0] for i in feasible}
        candidates[t.id] = ([i for i in fixed_candidates[t.id] if i in feasible][:budget] if fixed_candidates is not None
                            else sorted(feasible, key=lambda i: (scores[i],i))[:budget])
        for i in candidates[t.id]:
            nodes[t.id,i] = (t.work/max(1, snapshot.states["C"][snapshot.agent_compute[i]][-1][0]),
                             min(.999999, max(.000001, snapshot.states["R"][i][-1][0])))
    for a,b,mb in workflow.edges:
        a,b=int(a),int(b)
        for i in candidates[a]:
            for j in candidates[b]:
                path = snapshot.path(i,j)
                if path is None:
                    continue
                if path:
                    raw = np.asarray([snapshot.states["E"][p][-1] for p in path])
                    edges[a,b,i,j] = (float(8*mb/raw[:,1].min()+raw[:,0].sum()), float(np.prod(1-raw[:,2])))
                else:
                    edges[a,b,i,j] = (0.,1.)
    return Prediction(workflow.id,candidates,nodes,edges)


def observed_loss(model, example, sparse_weight=0.0, retrieval_weight=.1):
    """训练损失：只用真实发生过的观测作为标签，不伪造未执行的目标。"""
    forced = {(o.task,o.agent) for o in example.observations}
    forced |= {(o.target_task,o.target_agent) for o in example.observations if o.kind == "edge"}
    output = model(*example.inputs(), forced_pairs=forced)
    decoded=model.decode(output,*example.inputs())
    maps = {"node": {key:j for j,key in enumerate(output["pairs"])},
            "edge": {key:j for j,key in enumerate(output["edge_keys"])}}
    losses, positive = [], []
    for o in example.observations:
        if o.kind=="edge" and o.agent==o.target_agent:
            continue  # 确定性的同机传递不产生随机通信目标，跳过
        key = (o.task,o.agent) if o.kind == "node" else (o.task,o.target_task,o.agent,o.target_agent)
        if key not in maps[o.kind]:
            raise ValueError("observed action missing from training graph")
        pred = output[o.kind][maps[o.kind][key]]
        duration=decoded[o.kind][0][maps[o.kind][key]]
        success_logit=decoded[o.kind][2][maps[o.kind][key]]
        # 对数时长损失平衡长短任务；成功率用带 logits 的二元交叉熵
        losses.append(F.smooth_l1_loss(torch.log1p(duration), pred.new_tensor(math.log1p(o.duration)))
                      + F.binary_cross_entropy_with_logits(success_logit, pred.new_tensor(float(o.success))))
        if o.kind == "node" and o.success:
            # 检索对比损失：让真实选中的智能体在可行集合中得分最高
            feasible = example.snapshot.feasible(example.workflow.tasks[o.task])
            score = output["retrieval_scores"][o.task, feasible]
            positive.append(F.cross_entropy(score[None], torch.tensor([feasible.index(o.agent)],device=score.device)))
    zero = output["sparse_loss"] * 0
    prediction = torch.stack(losses).mean() if losses else zero
    retrieval = torch.stack(positive).mean() if positive else zero
    # 总损失 = 预测损失 + 检索损失权重 + 稀疏门控惩罚 + 辅助损失权重
    loss = prediction + retrieval_weight*retrieval + sparse_weight*output["sparse_loss"] + .05*output["aux_loss"]
    return loss, {"prediction": float(prediction.detach()), "retrieval": float(retrieval.detach()),
                  "observations": len(losses)}
