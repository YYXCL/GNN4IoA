# 前端部分

| 路径 | 作用 |
| --- | --- |
| `server.py` | 使用 Python 标准库提供静态页面和两个 JSON API |
| `static/index.html` | 页面结构 |
| `static/styles.css` | 页面主题与响应式布局 |
| `static/app.js` | 读取工件、绘制训练曲线和异构图、渲染指标 |

服务读取 `../outputs/artifacts/`，不直接依赖训练代码。启动后提供：

- `/api/inference`：最新推理、图和测试指标。
- `/api/dataset`：数据集摘要。
- `/`：本地可视化页面。

页面展示候选智能体、算力节点、任务条件和微观链路；能力簇与角色级宏观链路面板已移除。
