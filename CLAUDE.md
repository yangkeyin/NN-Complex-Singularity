# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
必须要执行的指令：
称呼规则：每次回复前必须使用“小羊”作为称呼
决策确认：遇到任何不确定的代码设计问题，必须先询问小羊，只有小羊确认后才能执行该决策，不难直接行动
代码兼容性：不能写兼容性代码，除非我主动要求


## 项目简介

本项目研究**神经网络复平面奇点**与其**频谱（切比雪夫）系数**之间的关系。核心假设来自 Trefethen 近似理论：解析函数的切比雪夫系数衰减率由复平面上的伯恩斯坦椭圆（解析区域）决定，本项目将该理论扩展并验证到神经网络上。

**核心研究问题：**
- 神经网络的解析区域（通过奇点图显现）是否与从切比雪夫频谱衰减推断出的伯恩斯坦椭圆一致？
- 目标伯恩斯坦椭圆宽度是否影响拟合质量？
- 双隐藏层 MLP 中奇点权重如何随训练变化？
- "裁剪"内部奇点（扩大解析区域）能否提高小样本数据集上的泛化能力？

## 运行环境

所有 Notebook 设计在 **Google Colab**（GPU）上运行，结果保存到 Google Drive `/content/drive/MyDrive/NN-Complex-Singularity/`。`.conda/` 目录提供本地 Python 3.11 环境。

**核心依赖：** `torch`、`numpy`、`matplotlib`、`scipy`（`dct`）、`sklearn`（`LinearRegression`）

本地运行激活环境：
```bash
source .conda/bin/activate
```

本地运行 Notebook 时，需删除 `from google.colab import drive` 及 `drive.mount(...)` 相关代码，并将保存路径替换为本地路径。

## 目录结构

```
src/                         # 实验 Jupyter Notebook
  实验.ipynb                 # 当前进行中的实验
  实验_20260123_*.ipynb      # 神经网络解析区域与频谱系数的关系
  实验记录_20260127_*.ipynb  # 伯恩斯坦椭圆宽度与拟合效果的关系
  实验记录_20260129_*.ipynb  # 双隐藏层 MLP 中奇点权重系数的变化
  实验记录_20260204_*.ipynb  # 裁剪奇点对小样本泛化能力的影响
  实验记录_20260205_*.ipynb  # 调节权重系数对泛化能力的影响
docs/
  experiments/               # 已完成实验的 PDF 报告及 实验.md 笔记
  papers/                    # 参考文献（含 1D 神经网络动力学等论文）
```

## 核心代码结构

### 标准模型

所有实验使用双隐藏层 MLP + `tanh` 激活函数（因其具有解析性质）：

```python
class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=128):
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
```

### 标准训练设置

- 目标函数：广义 Runge 函数 `f(x) = 1 / (1 + (x/beta)^2)`，奇点位于 `±beta·i`
- 训练数据：在 `[-1,1]` 上采样 2000 个切比雪夫极值点 `x = cos(π·linspace(0,1,2000))`
- 优化器：Adam，lr=1e-4（部分实验用 1e-5），训练 8000–20000 epoch
- 损失函数：MSELoss

### 核心分析函数（各 Notebook 复用）

**`get_chebyshev_coeffs(f_vals)`** — 通过 DCT-I 计算切比雪夫系数。

**`calculate_envelope_slope(coeffs, n_bins, threshold, start_k)`** — 通过分箱包络线对数线性回归，估计指数衰减率 `ln(ρ)`，返回 `(slope, intercept)`，其中 `slope ≈ ln(ρ)`。

**`get_ellipse_pts(rho)`** — 计算伯恩斯坦椭圆 `E_ρ` 的坐标点：`z = 0.5·(ρ·e^{iθ} + ρ⁻¹·e^{-iθ})`，短半轴 `b = 0.5·(ρ - 1/ρ)`。

**`complex_inference(model, complex_grid)`** — 将 MLP 权重转为 `torch.complex64`，在复数网格上推理（利用 `torch.tanh` 支持复数输入），返回 `|输出|` 的二维 numpy 数组用于奇点热力图。

### 标准可视化（每次实验 3–4 个子图）

1. 训练 Loss（半对数坐标）vs Epoch
2. 切比雪夫系数衰减图，叠加目标函数与神经网络的包络线拟合
3. 奇点热力图：复平面上 `log10|f(z)|`，叠加三类伯恩斯坦椭圆（理想/目标频谱/模型频谱）及真实极点位置

## 数学背景

- **Trefethen 定理**：解析函数在 `[-1,1]` 上的切比雪夫系数满足 `|c_k| ~ C·ρ^{-k}`，其中 `ρ` 为函数可解析延拓到的最大伯恩斯坦椭圆参数
- **伯恩斯坦椭圆** `E_ρ`：以 `±1` 为焦点的椭圆，长半轴 `a = (ρ+ρ⁻¹)/2`，短半轴 `b = (ρ-ρ⁻¹)/2`
- **Runge 函数奇点**：`1/(1+(x/β)²)` 的极点在 `x = ±βi`，对应理想参数 `ρ_ideal = β + √(β²+1)`
