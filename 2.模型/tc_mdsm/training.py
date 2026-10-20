"""训练、验证校准、模型保存与加载。"""
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import Predictor, observed_loss
from .types import ModelConfig
from .execution import execute_batch


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def calibrate(model,examples):
    """在验证集上做逻辑回归概率校准，修正成功率预测的偏差。"""
    model.eval();values=[[],[]];labels=[[],[]]
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
                values[kind].append(float(decoded[o.kind][2][maps[kind][key]]));labels[kind].append(float(o.success))
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
    model._state_cache=None
    model._index_cache=None



def evaluate_examples(model,examples,seed=7,time_limit=30.):
    rows=[]
    for e in examples:
        result,_,_=execute_batch(model,[e],model.config.candidate_budgets,time_limit,seed)
        rows.extend(result)
    return rows


def summary(rows):
    successful=[r["completion_s"] for r in rows if r["on_time"]]
    return {"on_time":float(np.mean([r["on_time"] for r in rows])),
            "penalized_s":float(np.mean([r["penalized_s"] for r in rows])),
            "completion_s":float(np.mean(successful)) if successful else None}


def train(train_cases,val_cases,output,method,config,seed=7,epochs=100,patience=10,lr=.001,
          device="cpu",resume=False,validation_limit=0,time_limit=30.):
    #训练样本集、验证样本集、输出目录、预测方法名、ModelConfig、全局随机种子、最大训练轮数、早停容忍度、Adam 优化器学习率、训练设备、是否从`last.pt` 断点续训、每轮只用验证集前 N 条评估、验证执行时 CP-SAT 调度求解的单样本时间上限
    output=Path(output);output.mkdir(parents=True,exist_ok=True)    # 创建输出目录 output。
    seed_everything(seed)   # 设置随机种子，保证可复现。
    model=Predictor(config,method).to(device);model.fit_normalizers(train_cases)    # 构建模型 Predictor并放到指定设备（CPU/GPU）。用训练数据拟合归一化器。
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)        # 定义优化器 Adam。lr是学习率
    first=0;history=[];best_score=(-1.,-float("inf"));stale=0   # 初始化训练状态：起始 epoch、历史记录、最佳分数、早停计数。
    last=output/"last.pt";best=output/"best.pt"                 # 定义保存路径 last.pt（最新模型）和 best.pt（最佳模型）。
    if resume and last.exists():        # 如果有断点就从断点处继续训练
        ck=torch.load(last,map_location=device,weights_only=False)
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"])
        first=ck["epoch"];history=ck["history"];best_score=tuple(ck["best_score"]);stale=ck["stale"]
    if method=="Rule-Latency":          # 如果 method == "Rule-Latency"，不进行梯度训练，直接保存模型并返回空历史。
        torch.save({"config":asdict(config),"method":method,"model":model.state_dict(),"seed":seed},best)
        return model,[]
    for epoch in range(first,epochs):   # 对每个 epoch：
        if stale>=patience and epoch>=7:break       # 早停判断：若 早停计数>= 10 且 epoch ≥ 7，则提前停止。
        started=time.perf_counter();model.train();model._state_cache=None;model._index_cache=None   # 模型开始训练
        order=list(range(len(train_cases)));random.shuffle(order);losses=[]     #打乱训练样本顺序。
        sparse_weight=.01*min(1.,max(0.,(epoch-1)/5))       #计算稀疏权重 sparse_weight（随 epoch 逐渐增大）
        for index in order:     # 对每个样本计算损失 (observed_loss函数在model.py文件里)。
            optimizer.zero_grad(set_to_none=True)
            loss,_=observed_loss(model,train_cases[index],sparse_weight)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)   # 反向传播、梯度裁剪
            optimizer.step();losses.append(float(loss.detach()))    # 更新参数
        calibrate(model,val_cases)      # 校准：用验证集调用 calibrate。
        selected=val_cases[:validation_limit] if validation_limit else val_cases    # 若设置了 validation_limit，只取前若干条验证样本。
        metrics=summary(evaluate_examples(model,selected,seed,time_limit))  # 调用evaluate_examples(第62行)得到指标，再用 summary(第70行)汇总。
        score=(metrics["on_time"],-metrics["penalized_s"])      # 评分标准：即准时率越高越好、惩罚时间越少越好。
        improved=score>best_score
        if improved:best_score=score;stale=0        # 若分数提升，更新 best_score
        else:stale+=1                               # 若分数没有提升，早停次数加1
        record={"epoch":epoch+1,"loss":float(np.mean(losses)),"seconds":time.perf_counter()-started,"validation":metrics}
        history.append(record)      # 保存当前 epoch 的 loss、耗时、验证指标。
        ck={"config":asdict(config),"method":method,"model":model.state_dict(),"seed":seed,
            "optimizer":optimizer.state_dict(),"epoch":epoch+1,"history":history,"best_score":best_score,"stale":stale}
        torch.save(ck,last)         # 保存检查点：始终保存最新模型
        if improved:torch.save(ck,best)     # 若当前最佳，额外保存 best.pt
        save_json(output/"history.json",history)        # 同时保存 history.json。
        print(f"{method} seed={seed} epoch={epoch+1}/{epochs} loss={record['loss']:.4f} val={metrics['on_time']:.3f} seconds={record['seconds']:.1f}",flush=True)       # 打印日志：输出 epoch、loss、验证准时率、耗时。
    ck=torch.load(best,map_location=device,weights_only=False)      # 训练结束后，加载 best.pt 中的最佳模型参数。
    model.load_state_dict(ck["model"]);model.eval();model._state_cache=None;model._index_cache=None     # 设为评估模式，清空缓存。
    return model,history        # 返回训练好的模型和完整历史记录。


def load_model(path,device="cpu"):
    ck=torch.load(path,map_location=device,weights_only=False)
    model=Predictor(ModelConfig(**ck["config"]),ck["method"]).to(device)
    model.load_state_dict(ck["model"]);model.eval()
    return model
