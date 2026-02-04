# IoV_LLM: Embodied AI for Integrated Sensing, Communication, Computation, and Control

### 项目简介
本项目旨在探索**具身智能体（Embodied AI Agent）**在自动驾驶与车联网（IoV）环境下的应用。针对 6G 时代“通感算控一体化”资源深度耦合且动态演进的特性，我们利用多模态大语言模型（MLLM）作为核心决策引擎，实现跨域资源的智能调度与协同控制。

### 核心技术点 (Technical Highlights)
* **具身智能体调度**：利用 LLM 的逻辑推理能力，在 O-RAN 架构下实现对感知、通信、计算与控制资源的毫秒级智能调度。
* **多模态感知融合**：整合车辆传感器数据，赋能 Agent 在动态环境下进行通感算控的一体化决策。
* **工程实践**：
    * **模块化开发**：采用 Python 进行核心算法实现，确保代码的高可扩展性。
    * **规范化管理**：通过严格的 Git 工作流与环境隔离（Conda/.gitignore），保证实验的可追溯性与环境一致性。

### 快速开始
```bash
# 克隆仓库
git clone git@github.com:1neilkou/IoV_LLM.git

# 运行基础训练脚本
python train_iov.py
```
