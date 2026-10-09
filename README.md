# TC-MDSM 实验代码

项目比较 TC-MDSM、GMATANN、HM-MATAS、Rule-Latency、Full-State、StaticGraph。统一入口为 `3.实验评估/tc_mdsm_v3_runner.py`。

| 目录 | 内容 |
|---|---|
| `1.数据` | 基础能力、任务、负载及数据处理脚本 |
| `2.模型/tc_mdsm` | 预测模型、训练、候选检索、CP-SAT 和仿真执行 |
| `3.实验评估` | 数据生成、训练、评估和四幅图 |
| `4.结果输出` | 实验参数、数据、权重、逐实例结果和图表 |
| `工具脚本` | 能力扩展与批量运行入口 |
| `项目文档` | 方法与实验说明 |

## 运行

先安装对应平台的 PyTorch，再安装 `requirements.txt`。本机使用 `.venv-tc/Scripts/python.exe`，其他平台使用自己的 Python 环境。

```powershell
& ./.venv-tc/Scripts/python.exe 3.实验评估/tc_mdsm_v3_runner.py all --device cuda
```

也可依次执行 `generate`、`train`、`evaluate`、`report`。各阶段传入相同的输出目录和实验参数；改变配置时使用新目录。训练续跑指定 `--resume`，评估跳过已存在实例。

主要参数：`--output --counts --seeds --epochs --instances --device --seconds --candidates --planning-batch-size --validation-limit --method --resume`。默认训练/验证/测试样本数为12000/1500/1500，五种子7/17/29/43/61，最多100轮，每条件每种子30批，规划预算30秒。默认规模耗时较长，缩减运行必须写清实际参数。

四幅图为任务完成率、成功请求平均完成时延、并发场景性能、系统规模性能，均输出 PNG/PDF。不同方法使用相同场景与环境种子。完成率包括拒绝、执行失败和迟到；时延只统计按期成功请求。执行不进行重规划。

详细设计见 `项目文档/当前基线与实验设计.md`。实验代码保持主流程，临时验证不写入项目，规则见 `AGENTS.md`。

## 在线调度更新

状态更新时调用`prepare_snapshot`提前编码并建立检索键；仿真在在线计时前执行此准备，未创建常驻后台线程。在线预算只包含任务预测、约束建模、求解与方案提取。默认`--candidates 8 --planning-batch-size 4`；分批规划会预留已得到方案的资源区间，各批分享剩余在线预算。

StaticGraph现在仅取消GRU，保留TC-MDSM门控；推理检索抽取被选中的维度进行内积，图预测主干仍保留隐藏维度。分段耗时写入已有实例JSON和逐批CSV，旧记录缺失的耗时不补造。

2026-10-09缩减实验使用旧实现，不能作为此次修改后的结果。此次未重训或重跑实验；StaticGraph旧权重没有按新门控训练，后续比较须重新训练并使用新的结果目录。详细变动见`项目文档/在线调度改动说明.md`。本仓库不上传本机`.venv-tc`或`4.结果输出`目录。
