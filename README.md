# 智能体—算力节点异构状态图（MA-TGN）

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

| 数据 | 文件 | 规模/说明 |
| --- | --- | --- |
| 智能体能力 | `1.data/3.智能体能力数据(包括智能体环境和知识库、API权限)/agent_capabilities.csv` | 120 个智能体，编号 1—120 |
| 智能体可靠性 | `1.data/4.智能体可靠性/agent_reliability.csv` | 与智能体编号一一对应 |
| 算力节点 | `1.data/1.负载数据/compute_node_inventory.csv` | 12 个虚拟节点，编号 C001—C012 |
| 节点负载 | `1.data/1.负载数据/compute_node_load_timeseries.csv` | 服务器级代理负载、内存和显存时序 |
| 通信链路 | `1.data/2.链路数据/communication_links.csv` | 480 条微观链路 |
| 任务 | `1.data/5.任务数据/tasks.csv` | 72 个任务，编号 T0001 起 |
| 事件 | `1.data/6.事件数据/event_stream.csv` | 360 条示例事件及稳定编码 |
| 生成说明 | `1.data/dataset_manifest.json` | 规模、假设、字段歧义解释和文件清单 |

能力与任务同时提供 JSON 版本，用于保留场景、源智能体类型、完整 skill 对象等结构化溯源信息。所有 CSV 使用 UTF-8 BOM，便于中文 Excel 与 Python 同时读取。

### 数据处理边界

- 源能力文件没有显式的应急救援标签。应急救援派生智能体只使用该文件已有的火灾、跌倒、入侵、告警、泄漏和交通事故等 skill。
- Alibaba trace 没有容器到物理服务器的映射，也没有 CPU 利用率。脚本以 `container_ip` 稳定哈希构造 12 个虚拟服务器节点；`server_load_rate_proxy` 是内存与估计显存利用率的最大值，不是 CPU 负载。
- 内存和显存总容量属于明确标注的仿真假设，原始使用率与使用量不会被冒充为真实物理服务器容量。
- 用户需求中链路第 5、6 列均写为“网络丢包”。第 6 列按模型既有链路状态定义为“网络抖动(ms)”。

## 模型训练数据与运行链路

现有 MA-TGN 训练入口仍读取事件序列 JSONL：

```text
1.data/数据集处理与生成/generate_synthetic_dataset.py
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

基础仿真 CSV/JSON 与训练 JSONL 已统一字段语义和事件编码，但当前训练器尚未直接将基础表转换为滑动时间窗样本；后续应增加这一 ETL 层，避免继续依赖完全合成的状态序列。

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

生成默认 MA-TGN JSONL 数据：

```powershell
python "1.data/数据集处理与生成/generate_synthetic_dataset.py" --agents 120 --compute-nodes 12 --samples 48
```

训练和评估：

```powershell
python "3.training/train.py" --epochs 5
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
