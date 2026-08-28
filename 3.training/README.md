# 模型训练部分

`train.py` 是训练、验证和最终测试的统一入口。

它默认读取 `../1.data/datasets/sixg_ma_state/`，组合候选分类损失、状态预测损失和教师分数损失，按验证集 Top-1 准确率保存最佳模型，并写出：

- `../outputs/checkpoints/model.pt`
- `../outputs/artifacts/training_history.json`
- `../outputs/artifacts/dataset_summary.json`
- `../outputs/artifacts/latest_inference.json`

数据缺失时，脚本会优先从 `../1.data/` 的智能体、可靠性、算力负载、链路和任务基础表自动重建滑动窗口 JSONL；只有基础表也不存在时才回退到合成数据。可用 `--no-generate-if-missing` 关闭自动生成。训练工件只输出单个智能体、算力节点和微观链路，不再生成能力簇或角色级宏观边。

基础表转换默认生成 72 个非重叠窗口，每个窗口覆盖 16 个源时间戳，并按时间顺序划分为 54/9/9 个 train/val/test 样本。任务的必需 Skill、输出格式、字段完整率和端到端时延分别存储在样本元数据中。
