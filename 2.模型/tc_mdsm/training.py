"""可复现的训练、验证集校准与可断点续训的检查点管理。"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_examples
from .model import Predictor,observed_loss
from .types import ModelConfig
from .execution import execute_batch
from .provenance import code_identity,digest


def seed_everything(seed):
    """统一设置 Python / NumPy / PyTorch 的随机种子，保证实验可复现。"""
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path,value):
    """原子地写 JSON：先写临时文件再替换，避免中途失败产生损坏文件。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    def serializable(item):
        if isinstance(item,np.generic):
            return item.item()
        if isinstance(item,np.ndarray):
            return item.tolist()
        raise TypeError(f"unsupported JSON value: {type(item)}")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=serializable),encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path,value):
    """原子地保存检查点（同样是临时文件 + 替换）。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    torch.save(value,tmp);tmp.replace(path)


def calibrate(model,examples):
    """在验证集上做逻辑回归概率校准，修正成功率预测的偏差。"""
    if any(e.provenance["split"]!="val" for e in examples):
        raise ValueError("calibration requires validation split")
    model.eval();values=[[],[]];labels=[[],[]];errors=[[],[]]
    with torch.no_grad():
        for e in examples:
            forced={(o.task,o.agent) for o in e.observations}
            forced|={(o.target_task,o.target_agent) for o in e.observations if o.kind=="edge"}
            out=model(*e.inputs(),forced_pairs=forced)
            decoded=model.decode(out,*e.inputs())
            maps=[{k:i for i,k in enumerate(out["pairs"])},{k:i for i,k in enumerate(out["edge_keys"])}]
            for o in e.observations:
                if o.kind=="edge" and o.agent==o.target_agent:
                    continue
                kind=int(o.kind=="edge")
                key=(o.task,o.agent) if kind==0 else (o.task,o.target_task,o.agent,o.target_agent)
                z=out[o.kind][maps[kind][key]]
                values[kind].append(float(decoded[o.kind][2][maps[kind][key]]));labels[kind].append(float(o.success))
                duration=float(decoded[o.kind][0][maps[kind][key]])
                errors[kind].append(abs(duration-o.duration))
    report={}
    for kind in range(2):
        if not values[kind]:
            continue
        x=torch.tensor(values[kind],device=model.device);y=torch.tensor(labels[kind],device=model.device)
        # 带正则的逻辑回归校准；用 Jeffreys 伪计数避免 0/1 概率。
        parameters=torch.tensor([1.,0.],device=model.device,requires_grad=True)
        optimizer=torch.optim.Adam([parameters],lr=.05)
        for _ in range(100):
            logits=x*parameters[0]+parameters[1]
            loss=F.binary_cross_entropy_with_logits(logits,y)+.001*(parameters[0]-1).square()
            loss+=.5/len(x)*(F.softplus(logits.mean())+F.softplus(-logits.mean()))
            optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            model.calibration[kind].copy_(parameters)
            probability=(x*parameters[0]+parameters[1]).sigmoid()
            report["node" if kind==0 else "edge"]={"count":len(x),"mae_s":float(np.mean(errors[kind])),
                  "brier":float(((probability-y)**2).mean()),"positive_fraction":float(y.mean())}
    model._state_cache=None
    return report


def evaluate_examples(model,examples,output=None,seed=7,time_limit=30.,budgets=(32,),mode="end_to_end",condition="main"):
    """对样本逐个执行调度+仿真，汇总结果行并可选地写出 JSON。"""
    rows=[];execution_logs=[];plan_logs=[]
    for index,e in enumerate(examples):
        result,events,plans=execute_batch(model,[e],budgets,time_limit,seed,mode)
        for row in result:
            # 标注方法名（含消融后缀）与条件，方便后续分组统计
            label=model.method if model.config.ablation=="none" else f"{model.method}[{model.config.ablation}]"
            row.update({"seed":seed,"method":label,"condition":condition,"mode":mode,"instance":index,
                        "trace_group":e.provenance["trace_group"],"split":e.provenance["split"]})
        rows.extend(result);execution_logs.extend(events)
        plan_logs.extend(asdict(p) for p in plans)
    if output:
        atomic_json(output,{"rows":rows,"executions":execution_logs,"plans":plan_logs})
    return rows


def summary(rows):
    """对结果行做汇总统计（准时率、完成率、惩罚时长、决策耗时等）。"""
    complete=[r["completion_s"] for r in rows if r["on_time"]]
    return {"instances":len(rows),"on_time":float(np.mean([r["on_time"] for r in rows])),
            "completed":float(np.mean([r["completed"] for r in rows])),
            "penalized_s":float(np.mean([r["penalized_s"] for r in rows])),
            "completion_s":float(np.mean(complete)) if complete else None,
            "decision_s":float(np.mean([r["decision_s"] for r in rows])),
            "decision_p95_s":float(np.percentile([r["decision_s"] for r in rows],95))}


def train(dataset,output,method,config,seed=7,epochs=100,patience=10,lr=1e-3,device="cpu",resume=False,
          validation_limit=0,time_limit=30.,train_cases=None,val_cases=None):
    """训练主流程：逐 epoch 优化，按验证指标保存最优模型，支持断点续训。"""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seed_everything(seed)
    manifest=json.loads((Path(dataset)/"manifest.json").read_text(encoding="utf-8"))
    # 运行身份指纹：数据、方法、配置、代码版本等任一变化都会导致指纹不同
    run_identity={"data":manifest,"method":method,"config":asdict(config),"seed":seed,"lr":lr,
                  "validation_limit":validation_limit,"solver_seconds":time_limit,"patience":patience,"code":code_identity()}
    fingerprint=digest(run_identity)
    train_cases=train_cases if train_cases is not None else list(read_examples(Path(dataset)/"train.jsonl.gz"))
    val_cases=val_cases if val_cases is not None else list(read_examples(Path(dataset)/"val.jsonl.gz"))
    model=Predictor(config,method).to(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model.fit_normalizers(train_cases)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    first,history,best_score,stale=0,[],(-1.,-float("inf")),0
    used_parameters=set()
    last=output/"last.pt";best=output/"best.pt"
    if resume and last.exists():
        # 续训：校验配置/数据/代码指纹一致，并恢复优化器与随机状态
        ck=torch.load(last,map_location=device,weights_only=False)
        if ck["fingerprint"]!=fingerprint:
            raise ValueError("resume configuration or dataset mismatch")
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"])
        first,history,best_score,stale=ck["epoch"],ck["history"],tuple(ck["best_score"]),ck["stale"]
        used_parameters=set(ck.get("used_parameters",[]))
        random.setstate(ck["rng_python"]);np.random.set_state(ck["rng_numpy"]);torch.set_rng_state(ck["rng_torch"].cpu())
        if device.startswith("cuda") and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
    if method=="Rule-Latency":
        # 规则基线无需训练，直接保存一个占位检查点
        atomic_checkpoint(best,{"config":asdict(config),"method":method,"model":model.state_dict(),"seed":seed,"fingerprint":fingerprint})
        return model,[]
    atomic_json(output/"run_manifest.json",run_identity)
    for epoch in range(first,epochs):
        if stale>=patience and epoch>=7:
            break     # 早停
        started=time.perf_counter();cpu_started=time.process_time();model.train();model._state_cache=None
        order=list(range(len(train_cases)));random.shuffle(order)
        losses=[]
        # 稀疏门控惩罚权重随训练逐步增大，前几个 epoch 先预热
        sparse_weight=.01*min(1.,max(0.,(epoch-1)/5))
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            loss,_=observed_loss(model,train_cases[index],sparse_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {method}/{epoch}/{index}")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            used_parameters.update(name for name,p in model.named_parameters() if p.grad is not None)
            optimizer.step();losses.append(float(loss.detach()))
        train_seconds=time.perf_counter()-started
        calibration=calibrate(model,val_cases)          # 概率校准
        from .gate_audit import gate_audit
        gating=gate_audit(model,val_cases)              # 门控审计
        evaluation_cases=val_cases[:validation_limit] if validation_limit else val_cases
        validation=evaluate_examples(model,evaluation_cases,seed=seed,time_limit=time_limit,budgets=config.candidate_budgets)
        metrics=summary(validation);score=(metrics["on_time"],-metrics["penalized_s"])
        improved=score>best_score
        if improved:
            best_score,stale=score,0
        else:
            stale+=1
        record={"epoch":epoch+1,"loss":float(np.mean(losses)),"train_seconds":train_seconds,
                "process_cpu_seconds":time.process_time()-cpu_started,
                "parameters_with_training_gradient":sum(p.numel() for name,p in model.named_parameters() if name in used_parameters),
                "total_seconds":time.perf_counter()-started,"validation":metrics,"calibration":calibration,
                "sparse_weight":sparse_weight,"gate_audit":gating,"peak_gpu_bytes":torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0}
        history.append(record)
        # 保存检查点（含 RNG 状态，保证续训完全可复现）
        ck={"config":asdict(config),"method":method,"model":model.state_dict(),"seed":seed,"fingerprint":fingerprint,
            "optimizer":optimizer.state_dict(),"epoch":epoch+1,"history":history,"best_score":best_score,"stale":stale,
            "used_parameters":sorted(used_parameters),
            "rng_python":random.getstate(),"rng_numpy":np.random.get_state(),"rng_torch":torch.get_rng_state(),
            "rng_cuda":torch.cuda.get_rng_state_all() if device.startswith("cuda") else None}
        atomic_checkpoint(last,ck)
        if improved:
            atomic_checkpoint(best,ck)
        atomic_json(output/"history.json",history)
        print(json.dumps({"method":method,"seed":seed,**record},ensure_ascii=False),flush=True)
    chosen=torch.load(best,map_location=device,weights_only=False)
    model.load_state_dict(chosen["model"]);model.eval();model._state_cache=None
    return model,history


def load_model(path,device="cpu"):
    """从检查点加载模型并切换到评估模式。"""
    ck=torch.load(path,map_location=device,weights_only=False)
    ck["config"].setdefault("prediction_mode","absolute")
    model=Predictor(ModelConfig(**ck["config"]),ck["method"]).to(device)
    model.load_state_dict(ck["model"]);model.eval()
    return model
