# 智能体—算力节点异构状态图（MA-TGN）

> V3 使用智能体、算力和瞬时任务三类节点。节点状态与链路状态分别经 GRU 编码历史，一层聚合后由每个子任务学习五类状态的动态权重及任务—智能体注意力，最后按 DAG 顺序累加为复杂任务分配结果。完整说明见 [`docs/V3_MODEL_AND_BASELINES.md`](docs/V3_MODEL_AND_BASELINES.md)。

当前 `main` 分支维护 V3。旧版设计说明仍保留用于复现；新增规则、MLP、静态图和最小状态消融 baseline 已分别放在 [`baselines/`](baselines/) 下的独立命名目录。最新六方法论文仿真见 [`docs/PAPER_SIMULATION_V4.md`](docs/PAPER_SIMULATION_V4.md)。

本项目面向 6G 多智能体协同调度，使用任务条件化异构时序图网络统一表示智能体、云/边算力节点、通信链路和任务条件。当前版本直接建模单个智能体及其微观连接，不进行智能体簇建模。

## 项目结构

```text
GNN4IoA/
├─ 1.data/                # 原始数据、基础仿真数据、训练集及数据生成脚本
│  ├─ 1.负载数据/          # 虚拟算力节点清单和服务器级负载时序
│  ├─ 2.链路数据/          # 智能体—智能体、智能体—算力节点链路
│  ├─ 3.智能体能力数据(包括智能体环境和知识库、API权限)/
│  ├─ 4.智能体可靠性/
│  ├─ 5.任务数据/
│  ├─ 6.事件数据/          # 事件/关系编码表和示例事件流
│  ├─ 实验室的智能体能力列表/
│  ├─ 数据集处理与生成/
│  ├─ ma_tgn_data/        # 仿真与 JSONL 序列化
│  ├─ datasets/           # 可直接用于训练的结构化数据集
│  └─ raw/                # Alibaba 原始 trace 与旧的容器级处理结果
├─ 2.模型/                # 状态编码、图记忆、注意力和 MA-TGN 模型
├─ 3.training/           # 训练、验证、测试与工件生成
├─ 4.前端/                # 本地可视化页面和服务
├─ 推理演示/              # 加载检查点进行 Top-K 推理
├─ baselines/            # 规则、MLP、静态图、特征消融及 GMATANN/GCAN 基线
├─ docs/                 # 模型、数据和实验说明
├─ experiments/          # 统一评估、版本与 baseline 对比
├─ outputs/              # 检查点和前端消费的运行工件
├─ tests/                # 自动化测试
├─ pyproject.toml
└─ requirements.txt
```

## 基础仿真数据

运行下面的命令，会从 24 类源能力卡中为每类派生 5 个智能体，并处理 Alibaba GenAI trace：

```powershell
python "1.data/数据集处理与生成/generate_simulation_datasets.py"
```

默认输出：

| 数据     | 文件                                                           | 规模/说明                             |
| ------ | ------------------------------------------------------------ | --------------------------------- |
| 智能体能力  | `1.data/3.智能体能力数据(包括智能体环境和知识库、API权限)/agent_capabilities.csv` | 120 个智能体，编号 1—120                 |
| 智能体可靠性 | `1.data/4.智能体可靠性/agent_reliability.csv`                      | 与智能体编号一一对应                        |
| 算力节点   | `1.data/1.负载数据/compute_node_inventory.csv`                   | 12 个虚拟节点，编号 C001—C012             |
| 节点负载   | `1.data/1.负载数据/compute_node_load_timeseries.csv`             | 服务器级代理负载、内存和显存时序                  |
| 通信链路   | `1.data/2.链路数据/communication_links.csv`                      | 7,260 条拓扑边 × 48 个动态观测点；智能体全连接且固定一对一接入算力节点 |
| 任务     | `1.data/5.任务数据/tasks.csv`                                    | 72 个复杂任务 + 351 个子任务；显式串并行 DAG |
| 事件     | `1.data/6.事件数据/event_stream.csv`                             | 360 条示例事件及稳定编码                    |
| 训练窗口   | `1.data/datasets/sixg_ma_state/*.jsonl`                      | 72 个由基础表转换的非重叠时序窗口，按时间划分训练/验证/测试集 |
| 生成说明   | `1.data/dataset_manifest.json`                               | 规模、假设、字段歧义解释和文件清单                 |

能力与任务同时提供 JSON 版本，用于保留场景、源智能体类型、完整 skill 对象等结构化溯源信息。所有 CSV 使用 UTF-8 BOM，便于中文 Excel 与 Python 同时读取。

### 数据处理边界

- 源能力文件没有显式的应急救援标签。应急救援派生智能体只使用该文件已有的火灾、跌倒、入侵、告警、泄漏和交通事故等 skill。
- Alibaba trace 没有容器到物理服务器的映射，也没有 CPU 利用率。脚本以 `container_ip` 稳定哈希构造 12 个虚拟服务器节点；`服务器负载率代理` 是内存与估计显存利用率的最大值，不是 CPU 负载。
- 同一链路的标称带宽在全部时间戳保持不变，并从 500/1000/2000/3000 Mbps 四档中确定；变化的是时延、丢包和抖动，不把标称带宽误写成实时吞吐量。
- 内存和显存总容量属于明确标注的仿真假设，原始使用率与使用量不会被冒充为真实物理服务器容量。
- 通信拓扑与任务依赖完全分离：通信边只表示可达性和动态 QoS，依赖关系只写入复杂任务 DAG。
- 模型只用 GPU 算力剩余量、内存剩余量、显存剩余量表示算力节点；任务只用 Skill id、时延和准确率编码。

## 模型训练数据与运行链路

MA-TGN 训练入口读取由基础仿真表自动转换的事件序列 JSONL：

```text
1.data/数据集处理与生成/generate_simulation_datasets.py
                    │ 生成基础 CSV/JSON
                    ▼
1.data/ma_tgn_data/foundation.py
                    │ 16 个源时间戳/窗口
                    │
                    ▼
1.data/datasets/sixg_ma_state/*.jsonl
                    │
                    ▼
3.training/train.py ─────────► outputs/checkpoints/model.pt
                    └────────► outputs/artifacts/*.json
                                         │
                                         ▼
                                  4.前端/server.py

outputs/checkpoints/model.pt + test.jsonl ─► 推理演示/demo.py
```

默认生成 72 个窗口，并以时间先后按 54/9/9 划分 train/val/test；窗口之间不重叠。每个子任务作为瞬时任务节点连接到所有候选智能体，Skill id 负责硬匹配，时延和准确率参与任务编码；输出与完整率只保留在元数据中。训练入口发现 JSONL 缺失时会从编号基础表自动重建。

## 环境准备

建议使用 Python 3.9 或更高版本：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 常用命令

生成基础仿真数据：

```powershell
python "1.data/数据集处理与生成/generate_simulation_datasets.py"
```

仅从已有基础表重建 MA-TGN JSONL：

```powershell
python "1.data/数据集处理与生成/build_training_jsonl_from_foundation.py"
```

训练和评估：

```powershell
python "3.training/train.py" --epochs 10 --max-train-cases 0
```

复现同一数据划分、同一随机种子的 V1/V2 对比：

```powershell
git worktree add "../GNN4IoA-v1" v1.0
python "experiments/compare_versions.py" --epochs 10 --seed 7 --max-train-cases 0
```

训练六种分配方法并生成场景×难度、图规模和中等任务并发四幅论文图：

```powershell
python "experiments/run_paper_simulation.py" --epochs 8 --seeds 7,17,29 --output "outputs/paper_simulation_v5"
```

运行消除唯一候选退化、监督全部 DAG 子任务的可辨识压力基准：

```powershell
python "1.data/数据集处理与生成/build_training_jsonl_from_foundation.py" --output "1.data/datasets/sixg_ma_state_stress" --candidate-count 48 --events 96 --neighbor-fanout 8
python "experiments/run_paper_simulation.py" --dataset "1.data/datasets/sixg_ma_state_stress" --epochs 6 --seeds 7,17,29 --output "outputs/hypothesis_stress_v1"
```

协议、候选可辨识性审计和实际结果见 `docs/HYPOTHESIS_DRIVEN_BENCHMARK.md`。

如需在三模型对比中也包含归档的 V1，使用上面建立的工作目录并启用 V1：

```powershell
python "experiments/compare_allocation_models.py" --epochs 10 --seed 7 --include-v1
```

运行推理演示：

```powershell
python "推理演示/demo.py" --top-k 5
```

启动本地可视化：

```powershell
python "4.前端/server.py"
```

然后访问 <http://127.0.0.1:8000>。

运行测试：

```powershell
python -m pytest -q
```

## 代码职责

- 修改基础表、Alibaba 聚合或生成规则：`1.data/数据集处理与生成/`。
- 修改 JSONL 序列化和仿真事件：`1.data/ma_tgn_data/`。
- 修改输入编码、归一化、图记忆或注意力：`2.模型/ma_tgn/`。
- 修改损失、优化器或训练流程：`3.training/train.py`。
- 修改展示字段：同时检查 `3.training/train.py` 的 `dashboard_payload()` 与 `4.前端/static/app.js`。
- 修改推理输出：`推理演示/demo.py`。
