# 推理部分

`demo.py` 加载 `../outputs/checkpoints/model.pt` 和测试集第一条样本，输出候选智能体的 Top-K 分数、五项状态预测和任务条件化模态权重。

该脚本只在终端打印 JSON，不修改数据集，也不生成新的模型文件。
