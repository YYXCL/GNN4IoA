# TC-MDSM 基础数据

当前实验读取 `1.算力负载数据/compute_node_load_timeseries.csv`、`3.智能体能力数据/agent_capabilities.json` 和 `5.任务数据/tasks.json`。`2.模型/tc_mdsm/data.py` 将这些基础输入构造成带状态历史、权限、拓扑与工作流的实验样本。新样本写入带日期的 `6.结果输出` 目录。

`原始数据/` 与 `智能体能力源数据/` 保留来源文件，`数据处理脚本/` 中的脚本用于重建基础表。原始 Alibaba trace 没有物理服务器映射；数据处理脚本以确定性规则构造虚拟算力节点。链路状态、智能体能力配置、工作流及执行结果中的仿真成分应与原始 trace 区分。

旧 MA-TGN 专用的 JSONL 训练集与序列化模块已清理。当前正式数据通过 `4.实验评估/tc_mdsm_v3_runner.py generate` 生成。
