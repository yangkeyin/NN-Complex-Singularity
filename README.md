## 1. 项目愿景 (Research Vision)
- **核心课题**：结合 Dynamics（动力学）理论优化大模型架构（如 HuatuoGPT-Vision）。
- **技术栈**：PyTorch, Multimodal LLMs, 动力学分析 (Analytic Continuation, Eigenvalues).
- **最终目标**：构建一个物理/数学规律驱动的高性能视觉语言模型，并保持完整的实验溯源。

## 2. 知识库索引 (Knowledge Index)
- **文献参考**：见 `/docs/papers/`。重点关注 `dynamics_summary.md` 中的解析延拓逻辑。
- **实验记录**：见 `/docs/experiments/`。记录了从飞书迁移过来的所有历史实验（目的、步骤、结果）。
- **核心代码**：位于 `/src/`。包含模型定义、训练脚本及对齐算法（DPO/GRPO）。

## 3. 协作模式 (Collaboration Mode)
- **角色定位**：你不仅是代码助手，更是我的科研合伙人。
- **推理逻辑**：在修改代码前，请先检索 `/docs/experiments/` 中的失败案例，避免重蹈覆辙。
- **数学对齐**：当涉及到模型初始化或 Loss 设计时，优先参考动力学文献中的数学约束。

## 4. 实验工作流 (Standard Workflow)
1. **方案讨论**：我会先在 NotebookLM 中消化文献，将结论同步到 `docs/theory/`。
2. **代码实现**：请你基于最新的理论文档，在本地 `/src/` 中实现或重构功能。
3. **本地运行**：直接在终端执行训练/评估脚本，并分析结果。
4. **归档记录**：实验完成后，我会将飞书的实时笔记转存至 `/docs/experiments/`，请你据此更新 `TODO.md`。