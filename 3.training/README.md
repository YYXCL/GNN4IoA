# 模型训练部分

`train.py` 是训练、验证和最终测试的统一入口。

它默认读取 `../1.data/datasets/sixg_ma_state/`，组合候选分类损失、状态预测损失和教师分数损失，按验证集 Top-1 准确率保存最佳模型，并写出：

- `../outputs/checkpoints/model.pt`
- `../outputs/artifacts/training_history.json`
- `../outputs/artifacts/dataset_summary.json`
- `../outputs/artifacts/latest_inference.json`

数据缺失时，脚本默认调用数据包生成一份合成数据；可用 `--no-generate-if-missing` 关闭。训练工件只输出单个智能体、算力节点和微观链路，不再生成能力簇或角色级宏观边。
