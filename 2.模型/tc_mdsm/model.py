"""TC-MDSM and five baselines, sharing prediction and scheduling interfaces."""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .types import GROUP_DIMS, METHODS, ModelConfig, Prediction
from .layers import HeterogeneousLayer
from .retrieval import CandidateIndex


def hash_text(text, dimensions=128):
    vector = np.zeros(dimensions, np.float32)
    text = text.lower().replace(" ", "")
    for n in (1, 2, 3):
        for p in range(max(0, len(text) - n + 1)):
            key = int.from_bytes(hashlib.blake2b(text[p:p+n].encode(), digest_size=8).digest(), "little")
            vector[key % dimensions] += 1 if key & 256 else -1
    return vector / max(1, float(np.linalg.norm(vector)))


class Predictor(nn.Module):
    def __init__(self, config=None, method="TC-MDSM"):
        super().__init__()
        self.config = config or ModelConfig()
        self.method = method
        if method not in METHODS:
            raise ValueError(method)
        d, cfg = self.config.hidden, self.config
        self.encoders = nn.ModuleDict({g: nn.Linear(n + 1, d) for g, n in GROUP_DIMS.items()})
        self.grus = nn.ModuleDict({g: nn.GRU(d, d, batch_first=True) for g in GROUP_DIMS})
        for g, n in GROUP_DIMS.items():
            self.register_buffer(f"mean_{g}", torch.zeros(n))
            self.register_buffer(f"std_{g}", torch.ones(n))
        self.register_buffer("normalizers_fitted", torch.tensor(False))
        self.task = nn.Sequential(nn.Linear(136, d), nn.GELU(), nn.LayerNorm(d))
        self.dag = nn.Sequential(nn.Linear(d * 3 + 2, d), nn.GELU(), nn.LayerNorm(d))
        self.group_gate = nn.Linear(d, 4)
        self.dim_gate = nn.Linear(d, 4 * d)
        self.key = nn.ModuleDict({g: nn.Linear(d, d) for g in GROUP_DIMS})
        self.query = nn.ModuleDict({g: nn.Linear(d, d) for g in GROUP_DIMS})
        self.pair = nn.Sequential(nn.Linear(d * 4, d), nn.GELU(), nn.LayerNorm(d))
        self.types = nn.Embedding(6, d)
        self.edge_input = nn.Sequential(nn.Linear(d * 2 + 5, d), nn.GELU())
        self.layers = nn.ModuleList(HeterogeneousLayer(d, cfg.heads, edge_updates=method == "HM-MATAS") for _ in range(cfg.layers))
        self.node_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.edge_head = nn.Sequential(nn.Linear(3 * d + 5, d), nn.GELU(), nn.Linear(d, 2))
        # GMATANN's conditional sequence VAE and local/global attention.
        self.cvae = nn.LSTM(d * 2, d, batch_first=True)
        self.posterior = nn.Linear(d, 2 * d)
        self.decoder = nn.GRU(d * 2, d, batch_first=True)
        self.local_attention = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.global_attention = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.fusion = nn.Linear(d * 2, d)
        self.attention_scale = nn.Parameter(torch.ones(2))
        self.register_buffer("calibration", torch.tensor([[1., 0.], [1., 0.]]))
        self._state_cache = None
        self._index_cache = None
        self._text_cache = {}
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def device(self):
        return self.calibration.device

    @property
    def sparse(self):
        return self.method == "TC-MDSM" and self.config.ablation != "no_gates"

    def fit_normalizers(self, examples):
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
        snapshot.validate()
        if not bool(self.normalizers_fitted):
            raise RuntimeError("fit training-only normalization before inference")
        cache_key = (id(snapshot), snapshot.revision, tuple(p._version for p in self.parameters()))
        if not self.training and self._state_cache is not None and self._state_cache[0] == cache_key:
            return self._state_cache[1]
        z, sequences = {}, {}
        temporal = self.method in ("TC-MDSM", "Full-State", "GMATANN") and self.config.ablation != "no_history"
        for g in GROUP_DIMS:
            x = torch.tensor(snapshot.states[g], dtype=torch.float32, device=self.device)
            missing = torch.tensor(snapshot.missing[g], dtype=torch.bool, device=self.device)
            x = (x - getattr(self, f"mean_{g}")) / getattr(self, f"std_{g}")
            x = torch.where(missing[..., None], torch.zeros_like(x), x)
            x = torch.cat([x, missing[..., None].float()], -1)
            sequence = F.gelu(self.encoders[g](x))
            sequences[g] = sequence
            z[g] = self.grus[g](sequence)[1][-1] if temporal else sequence[:, -1]
        if not self.training:
            self._state_cache = (cache_key, (z, sequences))
        return z, sequences

    def encode_tasks(self, workflow):
        vectors = []
        for t in workflow.tasks:
            text = "|".join([t.text, t.skill, *t.tools, *t.knowledge])
            if text not in self._text_cache:
                self._text_cache[text] = hash_text(text)
            values = [t.work / 800, *[v / n for v, n in zip(t.demand, (4, 64, 32))],
                      (workflow.deadline - workflow.release) / 240, workflow.min_reliability,
                      len(t.tools) / 5, len(t.knowledge) / 5]
            vectors.append(np.concatenate([self._text_cache[text], values]))
        base = self.task(torch.tensor(np.asarray(vectors), dtype=torch.float32, device=self.device))
        pre, suc = torch.zeros_like(base), torch.zeros_like(base)
        counts = base.new_zeros((len(base), 2))
        sizes = base.new_zeros((len(base), 2))
        if self.config.ablation != "no_dag":
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
        d = self.config.hidden
        a = self.group_gate(q).sigmoid()
        v = self.dim_gate(q).reshape(len(q), 4, d).sigmoid()
        penalty = a.mean() + v.mean()
        if not self.sparse:
            return torch.ones_like(a), torch.ones_like(v), penalty * 0
        if not self.training and not getattr(self,"_continuous_gates",False):
            active = (a >= self.config.gate_threshold).to(a.dtype)
            for k in range(len(q)):
                if active[k, [0, 1, 3]].sum() == 0:
                    group = [0, 1, 3][a[k, [0, 1, 3]].argmax().item()]
                    active[k, group] = 1
            mask = torch.zeros_like(v).scatter_(-1, v.topk(self.config.top_b, -1).indices, 1)
            return active, mask, penalty
        return a, v, penalty

    def forward(self, snapshot, workflow, budget=None, fixed_candidates=None, forced_pairs=None):
        def stamp():
            if self.device.type=="cuda" and not self.training:
                torch.cuda.synchronize(self.device)
            return time.perf_counter()
        started = stamp()
        z, sequence = self.encode_state(snapshot)
        q = self.encode_tasks(workflow)
        a, masks, sparse_loss = self.gates(q)
        mapping = torch.tensor(snapshot.agent_compute, device=self.device)
        keys = {g: self.key[g](z[g][mapping] if g == "C" else z[g]) for g in ("A", "C", "R")}
        queries = {g: self.query[g](q) * masks[:, gi] for g, gi in (("A", 0), ("C", 1), ("R", 3))}
        retrieval_scores = sum(a[:, gi, None] * (queries[g] @ keys[g].T) / math.sqrt(self.config.top_b)
                               for g, gi in (("A", 0), ("C", 1), ("R", 3))) if self.training else None
        encode_end = stamp()
        budget = budget or self.config.candidate_budgets[0]
        candidates = {}
        index_key=(id(snapshot),snapshot.revision,self.config.index,tuple(p._version for p in self.parameters()))
        if not self.training and self._index_cache is not None and self._index_cache[0]==index_key:
            indexes=self._index_cache[1]
        else:
            indexes = {g: CandidateIndex(keys[g], self.config.index, self.config.index_seed + gi)
                       for gi, g in enumerate(("A", "C", "R"))}
            if not self.training:
                self._index_cache=(index_key,indexes)
        for k, task in enumerate(workflow.tasks):
            feasible = snapshot.feasible(task)
            if fixed_candidates is not None:
                selected = [i for i in fixed_candidates[k] if i in feasible][:budget]
            else:
                selected_set = set()
                for g, gi in (("A", 0), ("C", 1), ("R", 3)):
                    if self.training or a[k, gi] > 0:
                        selected_set.update(indexes[g].search(queries[g][k], feasible, budget))
                predecessors = [int(p) for p, child, _ in workflow.edges if child == k]
                anchors = {i for p in predecessors for i in candidates[p]} & set(feasible)
                selected_set.update(anchors)
                ordered = sorted(selected_set)
                if ordered:
                    scores = sum(a[k, gi] * (keys[g][ordered] @ queries[g][k]) / math.sqrt(self.config.top_b)
                                 for g, gi in (("A",0),("C",1),("R",3))).detach()
                    # Re-rank using all available predecessor candidate paths, not a privileged label.
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
                selected = sorted(set(selected) | {i for kk, i in forced_pairs if kk == k and i in feasible})
            candidates[k] = selected
        retrieve_end = stamp()
        pairs = [(k, i) for k in candidates for i in candidates[k]]
        if not pairs:
            return {"pairs": [], "edge_keys": [], "node": q.new_zeros((0, 2)), "edge": q.new_zeros((0, 2)),
                    "candidates": candidates, "retrieval_scores": retrieval_scores,
                    "sparse_loss": sparse_loss, "aux_loss": q.sum() * 0, "timing": {"total": time.perf_counter()-started}}
        k_ids = torch.tensor([k for k, _ in pairs], device=self.device)
        i_ids = torch.tensor([i for _, i in pairs], device=self.device)
        c_ids = mapping[i_ids]
        selected_z = [a[k_ids, gi, None] * masks[k_ids, gi] * self.key[g](z[g][c_ids if g == "C" else i_ids])
                      for g, gi in (("A", 0), ("C", 1), ("R", 3))]
        h_pair = self.pair(torch.cat([q[k_ids], *selected_z], -1))
        aux_loss = q.sum() * 0
        if self.method == "GMATANN":
            seq = (sequence["A"][i_ids] + sequence["C"][c_ids] + sequence["R"][i_ids]) / 3
            cond = q[k_ids, None].expand(-1, seq.shape[1], -1)
            encoded = self.cvae(torch.cat([seq, cond], -1))[1][0][-1]
            mu, logvar = self.posterior(encoded).chunk(2, -1)
            logvar = logvar.clamp(-8, 5)
            latent = mu + torch.exp(.5 * logvar) * torch.randn_like(mu) if self.training else mu
            reconstruction = self.decoder(torch.cat([latent[:, None].expand_as(seq), cond], -1))[0]
            aux_loss = F.mse_loss(reconstruction, seq.detach()) + .01 * (-.5 * (1 + logvar - mu.square() - logvar.exp()).mean())
            h_pair = h_pair + latent
        # Canonical merged pair nodes; graph nodes are constructed once per whole DAG.
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
            for u, v, rel in ((k, pi[k,i], 0), (ai[i], pi[k,i], 2)):
                add(u, v, rel, zero); add(v, u, rel+1, zero)
        for i in agent_ids:
            add(ai[i], ci[snapshot.agent_compute[i]], 4, zero)
            add(ci[snapshot.agent_compute[i]], ai[i], 5, zero)
        for parent, child, mb in workflow.edges:
            parent, child = int(parent), int(child)
            for i in candidates[parent]:
                for j in candidates[child]:
                    path = snapshot.path(i, j)
                    if path is None:
                        continue
                    raw = np.asarray([snapshot.states["E"][p][-1] for p in path])
                    physical = [mb/40, float(raw[:,0].sum()) if path else 0,
                                float(raw[:,1].min()/2000) if path else 1,
                                float(1-np.prod(1-raw[:,2])) if path else 0,
                                float(raw[:,3].sum()) if path else 0]
                    edge_positions.append(len(features))
                    add(pi[parent,i], pi[child,j], 6, zero)
                    add(pi[child,j], pi[parent,i], 7, zero)
                    edge_keys.append((parent, child, i, j)); raw_features.append(physical);path_ids.append(path)
        if self.method == "HM-MATAS":
            # Global state and task-order nodes supplement the assignment graph.
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
        graph_end = stamp()
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
        node = self.node_head(x[offset:offset+len(pairs)])
        if edge_keys:
            left = torch.tensor([pi[a,i] for a,b,i,j in edge_keys], device=self.device)
            right = torch.tensor([pi[b,j] for a,b,i,j in edge_keys], device=self.device)
            edge = self.edge_head(torch.cat([x[left], x[right], ef[edge_positions], raw_tensor_features], -1))
        else:
            edge = q.new_zeros((0,2))
        finished=stamp()
        return {"pairs": pairs, "edge_keys": edge_keys, "node": node, "edge": edge,
                "candidates": candidates, "retrieval_scores": retrieval_scores,
                "sparse_loss": sparse_loss, "aux_loss": aux_loss, "gates": a.detach(),
                "timing": {"encode": encode_end-started, "retrieve": retrieve_end-encode_end,
                           "graph": graph_end-retrieve_end, "predict": finished-graph_end,
                           "total": finished-started}}

    def predict(self, snapshot, workflow, **kwargs):
        if self.method == "Rule-Latency":
            return rule_prediction(snapshot, workflow, **kwargs)
        self.eval()
        wall_start=time.perf_counter()
        with torch.no_grad():
            output = self(snapshot, workflow, **kwargs)
            decoded=self.decode(output,snapshot,workflow,calibrated=True)
            result = []
            for kind, keys in enumerate((output["pairs"], output["edge_keys"])):
                duration,probability,_=decoded["node" if kind==0 else "edge"]
                values=torch.stack([duration,probability],dim=-1).detach().cpu().tolist()
                result.append({tuple(k):tuple(v) for k,v in zip(keys,values)})
            prediction = Prediction(workflow.id, output["candidates"], *result, timing=output["timing"],
                                    diagnostics={"gates": output.get("gates", torch.empty(0)).cpu().tolist(),
                                                 "assignment_nodes":len(output["pairs"]),"assignment_edges":len(output["edge_keys"]),
                                                 "index":self.config.index,"eligible_counts":{t.id:len(snapshot.feasible(t)) for t in workflow.tasks},"candidate_counts":{k:len(v) for k,v in output["candidates"].items()}})
            prediction.timing["decode_and_calibrate"]=time.perf_counter()-wall_start-prediction.timing["total"]
            prediction.timing["total"]=time.perf_counter()-wall_start
            prediction.validate()
            return prediction

    def decode(self,output,snapshot,workflow,calibrated=False):
        """Shared physical priors plus learned corrections; deterministic self-transfers are exact."""
        result={}
        for kind,name in enumerate(("node","edge")):
            raw=output[name]
            priors=[]
            if name=="node":
                for k,i in output["pairs"]:
                    c=snapshot.agent_compute[i]
                    priors.append((workflow.tasks[k].work/max(1,float(snapshot.states["C"][c][-1][0])),
                                   float(snapshot.states["R"][i][-1][0])))
            else:
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
                duration=prior[:,0]*torch.exp(2*torch.tanh(raw[:,0]))
                logits=torch.logit(prior[:,1].clamp(1e-5,1-1e-5))+raw[:,1]
            else:
                duration=F.softplus(raw[:,0])*(10 if kind==0 else 1)+(.001 if kind==0 else 0)
                logits=raw[:,1]
            calibrated_logits=logits*self.calibration[kind,0]+self.calibration[kind,1] if calibrated else logits
            probability=calibrated_logits.sigmoid().clamp(1e-6,1-1e-6)
            if name=="edge":
                internal=torch.tensor([i==j for a,b,i,j in output["edge_keys"]],device=self.device,dtype=torch.bool)
                duration=torch.where(internal,torch.zeros_like(duration),duration)
                probability=torch.where(internal,torch.ones_like(probability),probability)
            else:
                duration=duration.clamp_min(.001)
            result[name]=(duration,probability,logits)
        return result


def rule_prediction(snapshot, workflow, budget=32, fixed_candidates=None, **kwargs):
    started = time.perf_counter()
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
    return Prediction(workflow.id,candidates,nodes,edges,{"total":time.perf_counter()-started})


def observed_loss(model, example, sparse_weight=0.0, retrieval_weight=.1):
    forced = {(o.task,o.agent) for o in example.observations}
    forced |= {(o.target_task,o.target_agent) for o in example.observations if o.kind == "edge"}
    output = model(*example.inputs(), forced_pairs=forced)
    decoded=model.decode(output,*example.inputs())
    maps = {"node": {key:j for j,key in enumerate(output["pairs"])},
            "edge": {key:j for j,key in enumerate(output["edge_keys"])}}
    losses, positive = [], []
    for o in example.observations:
        if o.kind=="edge" and o.agent==o.target_agent:
            continue  # Deterministic local handoff has no stochastic communication target.
        key = (o.task,o.agent) if o.kind == "node" else (o.task,o.target_task,o.agent,o.target_agent)
        if key not in maps[o.kind]:
            raise ValueError("observed action missing from training graph")
        pred = output[o.kind][maps[o.kind][key]]
        duration=decoded[o.kind][0][maps[o.kind][key]]
        success_logit=decoded[o.kind][2][maps[o.kind][key]]
        # Log-duration loss balances long/short tasks. No fabricated unexecuted targets.
        losses.append(F.smooth_l1_loss(torch.log1p(duration), pred.new_tensor(math.log1p(o.duration)))
                      + F.binary_cross_entropy_with_logits(success_logit, pred.new_tensor(float(o.success))))
        if o.kind == "node" and o.success:
            feasible = example.snapshot.feasible(example.workflow.tasks[o.task])
            score = output["retrieval_scores"][o.task, feasible]
            positive.append(F.cross_entropy(score[None], torch.tensor([feasible.index(o.agent)],device=score.device)))
    zero = output["sparse_loss"] * 0
    prediction = torch.stack(losses).mean() if losses else zero
    retrieval = torch.stack(positive).mean() if positive else zero
    loss = prediction + retrieval_weight*retrieval + sparse_weight*output["sparse_loss"] + .05*output["aux_loss"]
    return loss, {"prediction": float(prediction.detach()), "retrieval": float(retrieval.detach()),
                  "observations": len(losses)}
