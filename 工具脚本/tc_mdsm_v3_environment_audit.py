"""环境审计脚本（仅用于验证，不做方法排名）。

目的：在不训练任何模型的前提下，用规则基线（Rule-Latency）跑一批验证样本，
统计准入率、准时率、失败原因、候选耗时离散度、历史顺序敏感性，以及外部随机
种子带来的耗时波动，用于检查仿真环境是否合理且稳定。

注意：本审计只用于验证环境性质，不是方法对比实验。
"""
from pathlib import Path
import collections,copy,json,sys
ROOT=Path(__file__).resolve().parents[1]                 # 仓库根目录
sys.path.insert(0,str(ROOT/'2.模型'))                     # 让脚本能 import tc_mdsm 包
import numpy as np
from tc_mdsm.data import Foundation                       # 数据基座（读取原始数据、生成 episode）
from tc_mdsm.world import World                           # 确定性仿真世界（执行/传输）
from tc_mdsm.types import ModelConfig                     # 模型超参数配置
from tc_mdsm.model import Predictor                       # 预测模型（此处用 Rule-Latency 规则基线）
from tc_mdsm.execution import execute_batch               # 调度 + 离散事件仿真执行
from tc_mdsm.training import atomic_json                  # 原子写 JSON
f=Foundation(ROOT/'1.数据');m=Predictor(ModelConfig(),'Rule-Latency')   # 数据基座 + 规则基线（无需训练）
rows=[];durations=[];history_changes=[];noise=[];reasons=collections.Counter()   # 各统计量的累加容器
for i in range(30):
    # 取第 i 个验证样本（100 个智能体、中等难度）
    e=f.episode('val',50000+i,20260920,agents=100,difficulty='中等')[0]
    r,events,_=execute_batch(m,[e],time_limit=30);rows+=r          # 执行一次仿真，收集结果行
    reasons.update(x['failure_reason'] for x in events if not x['success'])   # 统计各类失败原因
    task=e.workflow.tasks[0];eligible=e.snapshot.feasible(task)    # 首个任务及其可行智能体集合
    # 同一任务在不同可行智能体上的耗时离散度（P90/P10 比值）
    samples=[World(e.snapshot,e.provenance['world_seed']).execute(e.workflow,0,a,0,0)[0] for a in eligible]
    if samples:durations.append(float(np.percentile(samples,90)/max(.001,np.percentile(samples,10))))
    if eligible:
        a=eligible[0];s=copy.deepcopy(e.snapshot);c=s.agent_compute[a]
        # 把算力历史状态倒序（当前值保持不变），检验执行耗时是否依赖历史顺序
        s.states['C'][c][:-1]=list(reversed(s.states['C'][c][:-1]))
        original=World(e.snapshot,e.provenance['world_seed']).execute(e.workflow,0,a,0,0)[0]
        changed=World(s,e.provenance['world_seed']).execute(e.workflow,0,a,0,0)[0]
        history_changes.append(abs(original-changed))              # 历史顺序改变引起的耗时差异
        # 换 8 个不同的世界种子重复执行，衡量外部随机性造成的耗时波动
        repeats=[World(e.snapshot,100+j).execute(e.workflow,0,a,0,0)[0] for j in range(8)]
        noise.append(float(np.std(repeats)))
# result 各字段含义：purpose 用途说明、requests 请求数、admitted 准入数、first_pass_on_time 首次即准时数、
# execution_failed_after_admission 准入后执行失败数、failure_reasons 失败原因分布、
# candidate_duration_p90_p10_mean 候选耗时 P90/P10 均值、same_current_reordered_history_absolute_duration_change_mean_s
# 历史顺序改变的绝对耗时差均值、exogenous_seed_duration_std_mean_s 外部种子导致的耗时标准差均值、limitations 局限说明
result={'purpose':'validation-only environment audit; not a method ranking experiment','requests':len(rows),'admitted':sum(r['initial_admitted'] for r in rows),'first_pass_on_time':sum(r['on_time'] for r in rows),'execution_failed_after_admission':sum(r['first_failure_s'] is not None for r in rows),'failure_reasons':dict(reasons),'candidate_duration_p90_p10_mean':float(np.mean(durations)),'same_current_reordered_history_absolute_duration_change_mean_s':float(np.mean(history_changes)),'exogenous_seed_duration_std_mean_s':float(np.mean(noise)),'limitations':'Synthetic half-second service integration and reserved link capacity. Noise statistic includes external dynamics and execution noise, not a causal variance decomposition.'}
# 原子落盘到正式实验输出目录，并打印结果
dest=ROOT/'6.结果输出'/'2026-09-27_TC-MDSM正式实验'/'环境审计.json';atomic_json(dest,result);print(json.dumps(result),flush=True)
