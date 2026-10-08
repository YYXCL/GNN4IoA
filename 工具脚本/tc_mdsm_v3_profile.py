"""GPU 性能实测脚本。

测量内容：单步训练耗时与显存峰值，以及大图（1600 智能体、困难难度）推理的
显存峰值与耗时；结果写入 6.结果输出/2026-09-27_TC-MDSM正式实验/GPU实测.json，
用于估算全量训练的可行性与资源需求。
"""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1]              # 仓库根目录
sys.path.insert(0,str(ROOT/'2.模型'))                  # 让脚本能 import tc_mdsm 包
# 先加载隔离的 GPU 版 PyTorch 轮子，再复用其中兼容的科学计算依赖。
import torch
sys.path.append(str(ROOT/'.venv-tc'/'Lib'/'site-packages'))
from tc_mdsm.data import Foundation                   # 数据基座
from tc_mdsm.model import Predictor,observed_loss     # 预测模型与训练损失
from tc_mdsm.types import ModelConfig                 # 模型超参数配置
from tc_mdsm.training import atomic_json              # 原子写 JSON
torch.set_num_threads(2)                              # 限制 CPU 线程数，避免干扰计时
f=Foundation(ROOT/'1.数据')                            # 加载数据基座
e=f.episode('train',321,agents=100,difficulty='中等')[0]   # 取一个训练样本用于计时
m=Predictor(ModelConfig()).to('cuda');m.fit_normalizers([e])   # 模型搬到 GPU，并用训练样本拟合标准化参数
opt=torch.optim.Adam(m.parameters(),lr=.001)          # Adam 优化器
times=[]
for k in range(12):
    # 先同步再计时一次完整的"前向 + 反向 + 更新"训练步（前 2 步作为预热不计入）
    torch.cuda.synchronize();start=time.perf_counter();m.train();opt.zero_grad();loss,_=observed_loss(m,e);loss.backward();opt.step();torch.cuda.synchronize()
    if k>=2:times.append(time.perf_counter()-start)
# result 字段：torch 版本、GPU 名称、隐层维度、候选预算、平均单步训练耗时、
# 训练峰值显存、按 12000 步估算的总训练耗时（不含验证）
result={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'hidden':128,'candidate_budget':32,'mean_train_step_s':sum(times)/len(times),'train_peak_bytes':torch.cuda.max_memory_allocated(),'projected_12000_epoch_s_excluding_validation':12000*sum(times)/len(times)}
del loss;opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();m.eval()   # 清理显存并重置峰值统计
big=f.episode('val',322,agents=1600,difficulty='困难')[0]   # 构造大图样本（1600 智能体、困难难度）
with torch.no_grad():pred=m.predict(*big.inputs(),budget=32)   # 只做推理，不计算梯度
result.update(large_graph_agents=1600,large_graph_peak_bytes=torch.cuda.max_memory_allocated(),large_graph_prediction_s=pred.timing['total'])   # 补充大图推理的显存峰值与总耗时
atomic_json(ROOT/'6.结果输出'/'2026-09-27_TC-MDSM正式实验'/'GPU实测.json',result);print(json.dumps(result),flush=True)   # 原子落盘并打印结果
