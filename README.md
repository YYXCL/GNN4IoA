# 智能体—算力节点异构状态图（MA-TGN）

本项目面向 6G 多智能体协同调度，使用任务条件化的异构时序图网络（TGN）统一表示智能体、云/边算力节点和它们之间的通信或历史协作关系。项目包含完整的合成数据生成、模型训练、Top-K 推理、本地可视化和自动测试流程。

本文档重点说明三件事：项目主体由哪些文件构成、哪些目录是运行后生成的、哪些缓存或工具残留不属于项目主体。

## 1. 整理后的项目结构

```text
第二版(gpt)(1)/
├─ src/ma_tgn/                  # 核心 Python 包：状态、仿真、编码器、TGN 和数据结构
│  ├─ __init__.py               # 对外导出公共类
│  ├─ attention.py              # 任务条件化关系注意力
│  ├─ capabilities.py           # 动态能力槽、能力拆分和逐能力可靠性
│  ├─ config.py                 # 模型维度、事件/关系类型和实验开关
│  ├─ dataset.py                # TrainingCase 与 JSONL 的转换及数据集生成入口
│  ├─ encoders.py               # 节点、边、时间和任务编码器
│  ├─ memory.py                 # 快/慢双时间尺度节点记忆
│  ├─ model.py                  # MA-TGN 主模型、事件处理、查询和 Top-K
│  ├─ simulator.py              # 6G 智能体—算力状态及任务样本仿真器
│  ├─ state_schema.py           # 状态字段、单位和归一化尺度的唯一契约
│  ├─ synthetic.py              # 小规模合成样本辅助函数，主要供测试使用
│  └─ types.py                  # 图事件、任务、查询图和训练样本数据类
├─ frontend/                    # 无框架的本地可视化页面
│  ├─ index.html                # 页面结构
│  ├─ styles.css                # 页面样式和响应式布局
│  └─ app.js                    # 读取工件、绘图并渲染指标
├─ tests/
│  └─ test_model.py             # 模型、状态契约、能力槽和置信度测试
├─ datasets/                    # 【生成产物】合成训练/验证/测试数据及智能体 profile
│  └─ sixg_ma_state/
│     ├─ train.jsonl
│     ├─ val.jsonl
│     ├─ test.jsonl
│     ├─ metadata.json
│     └─ profiles/agent_NNN.json
├─ checkpoints/                 # 【生成产物】训练检查点
│  └─ model.pt
├─ artifacts/                   # 【生成产物】训练与前端消费的 JSON 工件
│  ├─ dataset_summary.json
│  ├─ training_history.json
│  └─ latest_inference.json
├─ generate_dataset.py          # 命令行：生成数据集
├─ train.py                     # 命令行：训练、评估并生成前端工件
├─ demo.py                      # 命令行：加载检查点执行 Top-K 推理
├─ dashboard.py                 # 命令行：启动本地 HTTP 可视化服务
├─ pyproject.toml               # Python 包构建与基础依赖声明
├─ requirements.txt             # 直接运行和测试所需依赖
├─ .gitignore                   # 排除缓存、安装元数据和可选实验产物
└─ README.md                    # 项目说明
```

`profiles/` 默认有 100 个结构相同的 `agent_000.json` 至 `agent_099.json`，因此目录树只展示命名规则，不逐项列出。

## 2. 文件分类：主体、生成产物与无关文件

### 2.1 项目代码主体

以下内容应长期保留，它们共同定义项目逻辑：

- `src/ma_tgn/`：核心模型包；
- `generate_dataset.py`、`train.py`、`demo.py`、`dashboard.py`：四个用户入口；
- `frontend/`：可视化页面；
- `tests/`：自动测试；
- `pyproject.toml`、`requirements.txt`：环境和打包配置；
- `README.md`、`.gitignore`：说明和版本控制规则。

### 2.2 运行后才出现的文件

| 路径                                       | 由什么产生                                        | 用途                         | 可否删除                         |
| ---------------------------------------- | -------------------------------------------- | -------------------------- | ---------------------------- |
| `datasets/sixg_ma_state/train.jsonl`     | `generate_dataset.py`；数据缺失时也可由 `train.py` 触发 | 训练样本，默认占全部样本的 75%          | 可；删除后需重新生成才能训练               |
| `datasets/sixg_ma_state/val.jsonl`       | 同上                                           | 每轮验证和最佳检查点选择，默认占 12.5%     | 可；删除后需重新生成                   |
| `datasets/sixg_ma_state/test.jsonl`      | 同上                                           | 最终评估和 `demo.py` 推理输入       | 可；删除后需重新生成                   |
| `datasets/sixg_ma_state/metadata.json`   | 同上                                           | 数据规模、划分、状态字段、单位、归一化尺度和设计开关 | 可；训练会读取它，不能只删该文件后直接训练        |
| `datasets/sixg_ma_state/profiles/*.json` | 同上                                           | 每个智能体的角色、细粒度能力槽和逐能力可靠性     | 可；用于检查动态能力状态，重新生成即可恢复        |
| `checkpoints/model.pt`                   | `train.py`                                   | 验证集 Top-1 最优模型的配置与权重       | 可；删除后必须重新训练，`demo.py` 才能再次运行 |
| `artifacts/training_history.json`        | `train.py`                                   | 每轮损失及验证指标，供前端画训练曲线         | 可；重新训练可恢复                    |
| `artifacts/dataset_summary.json`         | `train.py`                                   | `metadata.json` 的前端副本      | 可；重新训练可恢复                    |
| `artifacts/latest_inference.json`        | `train.py`                                   | 测试样例的节点、边、注意力、预测和聚合结果      | 可；前端没有它时会提示先训练               |

仓库当前保留了上述产物作为可直接运行的示例。它们与研究内容有关，但不是源码；只需要发布“最小源码包”时可以整体删除 `datasets/`、`checkpoints/` 和 `artifacts/`。

### 2.3 与项目代码主体无关、已清理的文件

本次整理已移除以下内容：

- `__pycache__/`、`src/ma_tgn/__pycache__/`、`tests/__pycache__/`：Python 导入或执行时生成的 `.pyc` 字节码缓存；
- `.pytest_cache/`：运行 pytest 后生成的用例发现和状态缓存；
- `src/ma_tgn.egg-info/`：执行可编辑安装或包安装时由 setuptools 生成的元数据；
- `.agents/`：空的工具残留目录，不被任何代码读取；
- `.git/`：空且不是有效 Git 仓库的复制残留目录。

前三类缓存以后仍可能自动出现，`.gitignore` 已将它们排除。不要把真正包含版本历史的有效 `.git/` 当作缓存删除；本次移除的目录经检查为空。

## 3. 运行链路

```text
generate_dataset.py
        │
        ▼
datasets/sixg_ma_state/*.jsonl + metadata.json + profiles/
        │
        ▼
     train.py ───────────────► checkpoints/model.pt
        │
        └────────────────────► artifacts/*.json ─► dashboard.py + frontend/
                                      ▲
checkpoints/model.pt + test.jsonl ─► demo.py（终端输出 Top-K，不写文件）
```

一次训练样本 `TrainingCase` 包含：按时间排序的图事件、查询时刻的候选局部图、任务 QoS 与能力向量、最佳候选标签、五项候选状态监督值，以及仿真场景元数据。

## 4. 环境准备与快速运行

建议使用 Python 3.9 或更高版本。在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`.venv/` 是本地环境，不属于项目主体，已加入 `.gitignore`。

### 4.1 生成数据

```powershell
python generate_dataset.py --agents 100 --compute-nodes 12 --samples 48
```

主要参数：

| 参数                   |                      默认值 | 含义                   |
| -------------------- | -----------------------: | -------------------- |
| `--output`           | `datasets/sixg_ma_state` | 输出目录                 |
| `--agents`           |                    `100` | 智能体数；当前研究约束要求不少于 100 |
| `--compute-nodes`    |                     `12` | 云/边算力节点总数            |
| `--samples`          |                     `48` | 三个划分的总样本数            |
| `--update-ms`        |                    `100` | 虚拟状态探针更新间隔，ms        |
| `--seed`             |                      `7` | 随机种子                 |
| `--duration-minutes` |                     `30` | 元数据记录的仿真场景时长         |

默认按约 75% / 12.5% / 剩余样本划分训练、验证和测试集。JSONL 每一行是一个完整 `TrainingCase`，而不是一个单独节点或一条边。

### 4.2 训练与评估

```powershell
python train.py --epochs 5
```

主要参数：

| 参数                                                   |                      默认值 | 含义                        |
| ---------------------------------------------------- | -----------------------: | ------------------------- |
| `--dataset`                                          | `datasets/sixg_ma_state` | 包含 JSONL 和元数据的数据目录        |
| `--epochs`                                           |                      `5` | 训练轮数                      |
| `--lr`                                               |                  `0.001` | AdamW 学习率                 |
| `--seed`                                             |                      `7` | PyTorch 随机种子，也用于缺失数据的自动生成 |
| `--output`                                           |   `checkpoints/model.pt` | 最佳检查点输出位置                 |
| `--artifacts`                                        |              `artifacts` | 前端 JSON 工件目录              |
| `--generate-if-missing` / `--no-generate-if-missing` |                       启用 | 训练集缺失时是否自动生成默认数据          |
| `--max-train-cases`                                  |                     `12` | 每轮最多使用的训练样本；`0` 表示使用全部样本  |

损失由三部分组成：候选智能体交叉熵、五项状态预测 BCE（权重 0.40）和教师调度分数 MSE（权重 0.10）。每轮在验证集计算 Top-1 准确率和五项 MAE，并按验证集 Top-1 保存最佳模型。

默认只取 12 个训练窗口是为了快速验证闭环；正式实验建议：

```powershell
python train.py --epochs 20 --max-train-cases 0
```

快速样本得到的数字只能说明代码和数据链路可运行，不能直接作为论文结论。

### 4.3 推理演示

```powershell
python demo.py --top-k 5
```

`demo.py` 读取 `checkpoints/model.pt` 和测试集第一条样本，在终端输出：候选智能体 ID、调度分数、成功率、归一化执行时间、工具失败率、幻觉率、可执行性，以及任务条件化的模态权重。它只打印 JSON，不生成新文件。

可用 `--checkpoint`、`--dataset` 和 `--top-k` 指定其他检查点、测试 JSONL 或返回数量。

### 4.4 本地可视化

```powershell
python dashboard.py
```

浏览器访问 <http://127.0.0.1:8000>。可用 `--host` 和 `--port` 修改监听地址与端口。服务本身无第三方 Web 依赖，提供：

- `/` 及前端静态文件；
- `/api/inference`：读取 `artifacts/latest_inference.json`；
- `/api/dataset`：读取 `artifacts/dataset_summary.json`。

页面展示训练曲线、六项测试指标、候选智能体—算力局部图、图注意力、能力簇、云边资源、历史协作链路、任务 QoS 和当前建模约束。启动服务不会修改文件。

## 5. 核心建模内容

### 5.1 图节点与关系

```text
云侧算力节点 ── compute_access ──┐
边侧算力节点 ── compute_access ──┤→ 智能体候选节点
其他智能体 ── communication ─────┤
历史协作者 ── historical_collaboration ─┘
```

节点分为 `agent` 和 `compute`；关系分为 `compute_access`、`communication` 和 `historical_collaboration`。事件类型包括上下文、环境、算力、链路、任务结果和能力新增变化。

### 5.2 智能体状态

- 上下文负载：窗口 token 数、上下文占用率；
- 环境状态：工具/知识库 API 限流与请求占用、用户上下文、物理传感器和环境匹配；
- 最多 8 个细粒度能力槽：语义向量、有效位和使用次数；
- 每个能力独立维护：任务成功率、平均执行时间、工具调用失败率和幻觉率。

能力槽从 profile 初始化。任务反馈出现未匹配能力时新增槽；槽满后替换使用次数最少的能力。离线仿真使用可复现哈希嵌入；接入真实系统时应给 `CapabilityRegistry` 注入语义嵌入模型，以改善同义能力匹配。

### 5.3 算力节点状态

算力节点保存 GPU 总量/余量（TFLOPS）、显存总量/用量（GB）、显存—GPU 带宽总量/用量（Mbps）、队列长度、当前 batch 和最大 batch。

前六项默认参与调度。队列和 batch 会写入图、工件并在前端展示，但 `ModelConfig.use_queue_batch_in_decision=False` 时不进入决策编码器。可将其设为 `True` 做消融实验。

### 5.4 边状态

所有关系边只编码四个可观测量：`latency_ms`、`bandwidth_mbps`、`packet_loss_rate` 和 `jitter_ms`。云内、云间、端云或跨网关等路径类别只影响仿真分布，不作为训练特征，避免在已有完整链路遥测时重复编码先验。

### 5.5 置信度与归一化

上下文、环境、算力和链路使用随观测年龄衰减的置信度：

```text
有效置信度 = 上报置信度 × exp(-观测年龄秒数 / τ)
```

默认 `τ=2 s`。能力和逐能力可靠性是慢状态，不附加观测置信度。原始数据保留 token、ms、Mbps、TFLOPS 和 GB 等物理单位，进入编码器时才按照 `state_schema.py` 的尺度截断并归一化。

## 6. 源码文件详细说明

### 6.1 顶层入口

| 文件                    | 详细职责                                                    | 读取                            | 写入/输出                                          |
| --------------------- | ------------------------------------------------------- | ----------------------------- | ---------------------------------------------- |
| `generate_dataset.py` | 解析数据规模和仿真参数，调用数据层生成三份 JSONL、元数据和智能体 profile             | 命令行参数、`src/ma_tgn/dataset.py` | `datasets/`；终端打印元数据 JSON                       |
| `train.py`            | 顺序处理事件、查询候选图、组合三类损失、验证、保存最佳模型、最终测试并组装前端载荷               | 数据集目录                         | `checkpoints/model.pt`、`artifacts/*.json`、终端指标 |
| `demo.py`             | 从检查点恢复配置和权重，读取一条测试样本，执行 Top-K 调度与五项状态预测                 | 检查点、测试 JSONL                  | 仅终端 JSON                                       |
| `dashboard.py`        | 基于标准库 `ThreadingHTTPServer` 提供静态页面和两个 JSON API，包含路径越界保护 | `frontend/`、`artifacts/`      | HTTP 响应；不写磁盘                                   |

### 6.2 核心包 `src/ma_tgn/`

| 文件                | 主要内容                                                                  | 在流程中的位置                              |
| ----------------- | --------------------------------------------------------------------- | ------------------------------------ |
| `__init__.py`     | 汇总导出 `ModelConfig`、`CapabilityRegistry`、模型和四个数据类                      | 给顶层脚本提供稳定导入入口                        |
| `types.py`        | 定义 `GraphEvent`、`TaskRequirement`、`QueryGraph`、`TrainingCase`         | 统一仿真器、持久化和模型之间的数据形状                  |
| `state_schema.py` | 定义节点/算力类型、上下文/环境/算力/边/QoS 字段、物理单位、置信度适用范围和归一化上限                       | 全项目状态字段的唯一事实来源；修改字段应先改这里             |
| `config.py`       | 定义节点数、能力槽维度、隐藏层维度、注意力头数、记忆更新率、事件/关系词表和队列开关，并校验维度约束                    | 构造仿真器、编码器、记忆和模型                      |
| `capabilities.py` | 提供能力拆分与文本嵌入协议、哈希嵌入、启发式/LLM 拆分器、能力槽匹配、新增、最少使用淘汰、可靠性指数更新和 profile 保存    | 管理智能体的慢变能力状态                         |
| `simulator.py`    | 初始化角色、智能体状态和云/边算力；模拟状态漂移、链路质量、任务反馈、事件序列、候选局部图及监督标签                    | 为 `dataset.py` 生成可训练的 `TrainingCase` |
| `dataset.py`      | 在张量数据类和 JSON 字典间双向转换，逐行读写 JSONL；按划分运行仿真器并生成 `metadata.json` 和 profile | 数据生成与训练之间的持久化层                       |
| `encoders.py`     | 分别编码智能体、算力、边、时间和任务；计算置信度衰减；按任务条件融合模态；执行原始物理量归一化                       | 把异构原始状态映射到统一隐藏空间                     |
| `memory.py`       | 构造事件消息，以门控方式更新快记忆和慢记忆，并按查询时间读取节点状态                                    | 捕获短期突变和长期稳定特征                        |
| `attention.py`    | 实现分组 softmax 和按关系、任务条件化的多头图注意力                                        | 在查询局部图中聚合邻居信息并输出注意力权重                |
| `model.py`        | 串联编码器、双时间尺度记忆和关系注意力；处理连续事件；输出候选 logits/分数、五项状态预测、模态权重与 Top-K          | 训练和推理的核心模型                           |
| `synthetic.py`    | 封装小配置仿真器，快速产生单个合成样本                                                   | 自动测试和轻量调试辅助，不是正式数据入口                 |

### 6.3 前端与测试

| 文件                    | 说明                                                                |
| --------------------- | ----------------------------------------------------------------- |
| `frontend/index.html` | 定义训练、局部图、能力簇、云边算力、协作骨干、任务条件和模型约束七个展示区域                            |
| `frontend/styles.css` | 深色仪表盘主题、网格布局、指标卡、图例、进度条和窄屏适配                                      |
| `frontend/app.js`     | 请求推理工件，使用 Canvas 绘制训练曲线和异构图，并渲染指标、资源、能力簇和宏观链路                     |
| `tests/test_model.py` | 覆盖端到端输出形状与注意力归一化、快慢记忆差异、梯度传播、状态字段契约、能力槽淘汰、置信度衰减，以及队列/batch 默认屏蔽约束 |

## 7. 数据与输出字段

### 7.1 JSONL 样本顶层字段

- `events`：事件流；每个事件含源/目标节点、节点类型、时间、事件/关系类型、两端状态、置信度和四维边状态；
- `graph`：查询局部图，含候选 ID、边索引、边状态、关系类型、边时间、边置信度和查询时间；
- `task`：8 维 QoS 需求和细粒度能力语义向量；
- `label`：正确候选在候选列表中的索引；
- `candidate_targets`：每个候选的五项监督值；
- `teacher_scores`：仿真器产生的教师调度分数；
- `metadata`：任务类型、候选角色、算力挂载和场景状态等辅助信息。

五项状态预测依次为：任务成功率、归一化执行时间、工具调用失败率、幻觉率、环境可执行性。

### 7.2 检查点

`model.pt` 是 PyTorch 序列化字典，只包含：

- `config`：创建模型所需的 `ModelConfig` 字段；
- `model`：最佳模型的 `state_dict`。

它不包含优化器状态，因此适合推理或重新加载权重，不支持严格从中断处继续同一优化器训练。

### 7.3 训练指标

- `top1_accuracy`：最佳候选预测正确率，越高越好；
- `task_success_rate_mae`：成功率平均绝对误差；
- `execution_time_mae`：归一化执行时间平均绝对误差；
- `tool_failure_rate_mae`：工具失败率平均绝对误差；
- `hallucination_rate_mae`：幻觉率平均绝对误差；
- `executability_mae`：环境可执行性平均绝对误差。

除准确率外，MAE 均越低越好。

## 8. 测试与清理

运行全部测试：

```powershell
python -m pytest -q
```

测试后出现 `.pytest_cache/` 和 `__pycache__/` 是正常现象，不应提交。需要恢复“仅源码”的最小目录时，可删除以下可再生内容：

```powershell
Remove-Item -Recurse -Force .pytest_cache, __pycache__, src/ma_tgn/__pycache__, tests/__pycache__ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force src/ma_tgn.egg-info -ErrorAction SilentlyContinue
```

若同时删除 `datasets/`、`checkpoints/` 和 `artifacts/`，请按“生成数据 → 训练 → 推理/前端”的顺序重新生成。

## 9. 修改项目时的建议顺序

1. 增删状态字段：先改 `state_schema.py`，再同步 `simulator.py` 和 `encoders.py`，最后补测试；
2. 修改模型尺寸或开关：改 `config.py`，旧检查点可能因维度变化而不能加载；
3. 修改 JSONL 结构：同时修改 `case_to_dict()` 与 `case_from_dict()`，并重新生成数据；
4. 修改前端数据字段：同步 `train.py` 的 `dashboard_payload()` 与 `frontend/app.js`；
5. 改动核心逻辑后运行 `python -m pytest -q`，再用少量样本训练验证完整链路。

