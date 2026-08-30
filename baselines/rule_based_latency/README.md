# Rule-Latency

先做精确 Skill id 匹配，再用上一时刻的链路时延、GPU 剩余量和未完成任务数估算完成时间，选择估计最短者。该方法无可训练参数；`train.py` 只生成实验协议检查点，`evaluate.py` 执行评估。
