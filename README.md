# TC-MDSM 实验代码

本项目包括 TC-MDSM 及论文中的五个基线：GMATANN、HM-MATAS、Rule-Latency、Full-State、StaticGraph。Full-State 与 TC-MDSM 使用相同的历史编码、任务查询、图预测、32候选预算和 CP-SAT 后端，仅取消状态组门控及维度掩码。StaticGraph与 TC-MDSM相比只消去历史编码部分。模型实现在 `2.模型/tc_mdsm/`，统一实验入口为 `4.实验评估/tc_mdsm_v3_runner.py`。

## 目录

| 路径              | 用途                      |
| --------------- | ----------------------- |
| `1.数据/`         | 源能力、任务、负载、链路及基础数据生成脚本   |
| `2.模型/tc_mdsm/` | 当前模型、五个基线共用的预测与调度代码     |
| `3.实验评估/`       | 数据生成、训练、冻结、评估和四项论文图     |
| `4.结果输出/`       | 带日期的历史结果与新运行结果          |
| `工具脚本/`         | 能力配置生成、环境审计、GPU 画像和批量运行 |
| `自动化测试/`        | 当前协议的验证测试               |

## 运行

先按平台安装合适的 PyTorch，再安装 `requirements.txt` 中的依赖。从项目根目录依次运行：

```powershell
python "4.实验评估/tc_mdsm_v3_runner.py" generate
python "4.实验评估/tc_mdsm_v3_runner.py" train
python "4.实验评估/tc_mdsm_v3_runner.py" freeze
python "4.实验评估/tc_mdsm_v3_runner.py" evaluate
python "4.实验评估/tc_mdsm_v3_runner.py" report
```

默认新结果写入 `4.结果输出/2026-09-27_TC-MDSM正式实验/运行/`。五个基线名称在图例中不加 adapted。

出图顺序：实验一任务完成率、实验二平均完成时延为柱状图；实验三并发场景、实验四系统规模为双指标曲线图。每项保存 PNG 和 PDF。实验协议及结果边界见 `项目文档/TC-MDSM正式实验说明.md`。
