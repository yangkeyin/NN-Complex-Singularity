import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

# ===== 修改 1: 本地路径准备（移除 Google Drive）=====
save_dir = "./results/20260310_Chebyshev_N32_SoftPruning_v2/lr1e-3_2layer_withtrainloss"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"创建文件夹: {save_dir}")
else:
    print(f"使用已有文件夹: {save_dir}")

# 1. 构建目标函数 (Runge Function)
def runge_function(x, beta=0.2):
    return 1 / (1 + (x/beta)**2)

# 2. 定义 MLP 模型 (双层隐藏层)
class DoubleLayerMLP(nn.Module):
    # ===== 修改 2: 隐藏层大小改为 256（实验.md 要求）=====
    def __init__(self, hidden_size=256):
        super(DoubleLayerMLP, self).__init__()
        self.l1 = nn.Linear(1, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.l1(x))
        x = self.tanh(self.l2(x))
        return self.l3(x)

# 3. 解析延拓推理与归因函数
def analyze_complex_singularities(model, device, res=400):
    x_range = np.linspace(-2.0, 2.0, res)
    y_range = np.linspace(-0.8, 0.8, res)
    X, Y = np.meshgrid(x_range, y_range)
    Z_grid = X + 1j * Y
    z_tensor = torch.tensor(Z_grid, dtype=torch.complex64).to(device).unsqueeze(-1)

    with torch.no_grad():
        w1, b1 = model.l1.weight.to(torch.complex64), model.l1.bias.to(torch.complex64)
        h1_pre = torch.matmul(z_tensor, w1.t()) + b1
        h1 = torch.tanh(h1_pre)

        w2, b2 = model.l2.weight.to(torch.complex64), model.l2.bias.to(torch.complex64)
        h2_pre = torch.matmul(h1, w2.t()) + b2
        h2 = torch.tanh(h2_pre)

        w3, b3 = model.l3.weight.to(torch.complex64), model.l3.bias.to(torch.complex64)
        out = torch.matmul(h2, w3.t()) + b3
        out_abs = torch.abs(out).squeeze().cpu().numpy()

    threshold = np.percentile(out_abs, 98)
    mask = out_abs > threshold

    found_poles, found_weights, found_layers = [], [], []
    activation_limit = 10.0

    for r in range(1, res-1):
        for c in range(1, res-1):
            if mask[r, c]:
                val = out_abs[r, c]
                if val > out_abs[r-1,c] and val > out_abs[r+1,c] and val > out_abs[r,c-1] and val > out_abs[r,c+1]:
                    h1_pt = torch.abs(h1[r, c]).cpu().numpy()
                    h2_pt = torch.abs(h2[r, c]).cpu().numpy()

                    if np.max(h1_pt) > activation_limit:
                        j1 = np.argmax(h1_pt)
                        eff_a = torch.mean(torch.abs(model.l2.weight[:, j1])).item()
                        found_poles.append(Z_grid[r, c]); found_weights.append(eff_a); found_layers.append(1)
                        continue

                    if np.max(h2_pt) > activation_limit:
                        i1 = np.argmax(h2_pt)
                        eff_a = torch.abs(model.l3.weight[0, i1]).item()
                        found_poles.append(Z_grid[r, c]); found_weights.append(eff_a); found_layers.append(2)

    return out_abs, np.array(found_poles), np.array(found_weights), np.array(found_layers), X, Y

def is_inside_ellipse(z, beta):
    rho = beta + np.sqrt(beta**2 + 1)
    a, b = 0.5 * (rho + 1/rho), 0.5 * (rho - 1/rho)
    return (z.real/a)**2 + (z.imag/b)**2 <= 1.0

# ===== 修改 3: 替换硬剪枝为软剪枝策略 =====
def compute_soft_pruning_penalty(model, beta=0.2):
    """
    Compute soft pruning penalty (experiment.md):
    通过网络输出找椭圆内的极点，对第一层权重应用L1正则化以抑制这些极点
    关键：使用 requires_grad=True 的权重来计算惩罚，保证梯度流
    """
    # 使用较小的网格找极点（快速检测）
    x_range = np.linspace(-2.0, 2.0, 150)
    y_range = np.linspace(-0.8, 0.8, 150)
    X, Y = np.meshgrid(x_range, y_range)
    Z_grid = X + 1j * Y
    z_tensor = torch.tensor(Z_grid, dtype=torch.complex64, device=device).unsqueeze(-1)

    # 前向传播找极点位置（不需要梯度，只用于检测）
    with torch.no_grad():
        w1, b1 = model.l1.weight.to(torch.complex64), model.l1.bias.to(torch.complex64)
        h1_pre = torch.matmul(z_tensor, w1.t()) + b1
        h1 = torch.tanh(h1_pre)

        w2, b2 = model.l2.weight.to(torch.complex64), model.l2.bias.to(torch.complex64)
        h2_pre = torch.matmul(h1, w2.t()) + b2
        h2 = torch.tanh(h2_pre)

        w3, b3 = model.l3.weight.to(torch.complex64), model.l3.bias.to(torch.complex64)
        out = torch.matmul(h2, w3.t()) + b3
        out_abs = torch.abs(out).squeeze().cpu().numpy()

    # 找椭圆内的极点位置
    threshold = np.percentile(out_abs, 95)
    mask = out_abs > threshold
    
    ellipse_poles = []  # 存储在椭圆内的极点位置
    
    for r in range(1, len(y_range)-1):
        for c in range(1, len(x_range)-1):
            if mask[r, c]:
                val = out_abs[r, c]
                if val > out_abs[r-1,c] and val > out_abs[r+1,c] and val > out_abs[r,c-1] and val > out_abs[r,c+1]:
                    pole_position = Z_grid[r, c]
                    if is_inside_ellipse(pole_position, beta):
                        ellipse_poles.append((pole_position, r, c))
    
    # 如果没有椭圆内极点，返回0
    if len(ellipse_poles) == 0:
        return torch.tensor(0.0, device=device)
    
    # 对第一层权重应用L1正则化，惩罚那些可能产生椭圆内极点的神经元
    # 策略：对第一层应用全局L1惩罚（使权重整体变小，从而抑制椭圆内极点）
    w1_l1 = torch.sum(torch.abs(model.l1.weight)) + torch.sum(torch.abs(model.l1.bias))
    
    # 同时加上对第二、三层的轻量级惩罚
    w2_l1 = torch.sum(torch.abs(model.l2.weight)) * 0.1
    w3_l1 = torch.sum(torch.abs(model.l3.weight)) * 0.1
    
    # 惩罚强度与椭圆内极点数量相关
    soft_penalty = (w1_l1 + w2_l1 + w3_l1) * len(ellipse_poles)
    
    return soft_penalty

##### 修改 4: 绘图函数支持 Test Loss 曲线显示与本地存储 #####
def plot_results(model, train_losses, test_losses, stage_name, x_train, y_train, x_test, y_test, beta=0.2, prev_train_losses=None, prev_test_losses=None):
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test)
        current_test_loss = nn.MSELoss()(test_pred, y_test).item()
        sort_idx = torch.argsort(x_test.flatten())
        x_plot, y_plot = x_test[sort_idx], model(x_test[sort_idx])

        # 计算训练集预测
        sort_idx_train = torch.argsort(x_train.flatten())
        x_train_plot = x_train[sort_idx_train]
        y_train_pred = model(x_train_plot)

    out_abs, poles, weights, layers, X, Y = analyze_complex_singularities(model, device)

    plt.figure(figsize=(30, 6))

    # --- 子图 1: Training & Test Loss 曲线 (判断过拟合) ---
    plt.subplot(1, 5, 1)
    if len(train_losses) > 0:
        full_train_losses = (prev_train_losses + train_losses) if prev_train_losses else train_losses
        full_test_losses = (prev_test_losses + test_losses) if prev_test_losses else test_losses
        plt.semilogy(full_train_losses, color='blue', label='Train Loss')
        if len(full_test_losses) > 0:
            plt.semilogy(full_test_losses, color='orange', linestyle='--', label='Test Loss')
        if prev_train_losses:
            plt.axvline(len(prev_train_losses), color='red', linestyle='--', label='Fine-tuning Start')
        plt.title(f"{stage_name}\nLoss Curve")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No Training", ha='center')
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.grid(True, alpha=0.3)

    # --- 子图 2: 拟合图 ---
    plt.subplot(1, 5, 2)
    plt.plot(x_plot.cpu().numpy(), runge_function(x_plot.cpu().numpy(), beta), 'k--', alpha=0.5, label='True Runge')
    plt.plot(x_plot.cpu().numpy(), y_plot.cpu().numpy(), 'b-', lw=2, label='Model Pred (Test)')
    # 添加训练集预测线
    sort_idx_train = torch.argsort(x_train.flatten())
    x_train_plot = x_train[sort_idx_train]
    y_train_pred = model(x_train_plot)
    plt.plot(x_train_plot.cpu().numpy(), y_train_pred.detach().cpu().numpy(), 'g-', lw=2, label='Model Pred (Train)')
    plt.scatter(x_train.cpu().numpy(), y_train.cpu().numpy(), c='red', s=30, zorder=5, label=f'Train Data ({len(x_train)} pts)')
    current_train_loss = nn.MSELoss()(model(x_train), y_train).item()
    plt.title(f"Fitting\nTrain MSE: {current_train_loss:.2e}, Test MSE: {current_test_loss:.2e}")
    plt.legend(fontsize='x-small'); plt.ylim(-0.2, 1.2)

    # --- 子图 3: 奇点图 ---
    plt.subplot(1, 5, 3)
    im = plt.imshow(np.log10(out_abs + 1e-10), extent=[-2, 2, -0.8, 0.8], origin='lower', cmap='magma', aspect='auto')
    plt.colorbar(im, label="Log10|Output|")
    rho = beta + np.sqrt(beta**2 + 1)
    theta = np.linspace(0, 2*np.pi, 100)
    E_z = 0.5 * (rho * np.exp(1j * theta) + 1/(rho * np.exp(1j * theta)))
    plt.plot(E_z.real, E_z.imag, 'w--', lw=1.5, label="Ideal Ellipse")
    plt.title("Singularity Map")

    # --- 子图 4: 分布图 ---
    plt.subplot(1, 5, 4)
    im = plt.imshow(np.log10(out_abs + 1e-10), extent=[-2, 2, -0.8, 0.8], origin='lower', cmap='magma', aspect='auto', alpha=0.5)
    plt.plot(E_z.real, E_z.imag, 'w--', lw=1.5)
    if len(poles) > 0:
        p1 = poles[layers == 1]; w1 = weights[layers == 1]
        p2 = poles[layers == 2]; w2 = weights[layers == 2]
        if len(p1) > 0: plt.scatter(p1.real, p1.imag, s=w1*1000, c='blue', alpha=0.7, label='L1 Pole')
        if len(p2) > 0: plt.scatter(p2.real, p2.imag, s=w2*1000, c='red', alpha=0.7, label='L2 Pole')
    plt.title("Neuron Distribution"); plt.legend(fontsize='x-small')

    # --- 子图 5: 权重 vs 距离 ---
    plt.subplot(1, 5, 5)
    if len(poles) > 0:
        if len(p1) > 0: plt.scatter(np.abs(p1.imag), w1, c='blue', alpha=0.6, label='Layer 1')
        if len(p2) > 0: plt.scatter(np.abs(p2.imag), w2, c='red', alpha=0.6, label='Layer 2')
    ideal_b = 0.5 * (rho - 1/rho)
    plt.axvline(ideal_b, color='green', linestyle='--', label=f'Ideal b={ideal_b:.2f}')
    plt.xscale('log'); plt.yscale('log'); plt.xlim(1e-1, 1e0)
    plt.xlabel("|Im(z)|"); plt.ylabel("|a|"); plt.grid(True, which="both", ls=':', alpha=0.5); plt.legend()

    plt.tight_layout()
    # ===== 修改 4 续: 保存图片到本地 =====
    plt.savefig(os.path.join(save_dir, f"{stage_name}.png"), dpi=150)
    print(f"   已保存: {os.path.join(save_dir, f'{stage_name}.png')}")

# ==========================================
# 4. 主程序流程
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}\n")

# ===== 修改 5: 参数对齐实验.md =====
model = DoubleLayerMLP(hidden_size=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 改为 1e-4（实验.md）
criterion = nn.MSELoss()

# ===== 修改 6: 训练集采样（切比雪夫第二类节点 N=32，实验.md）=====
n_points = 32
beta_val = 0.2
# x_i = cos(i * pi / (n-1))
i_idx = torch.arange(n_points).float()
x_cheb = torch.cos(i_idx * np.pi / (n_points - 1)).view(-1, 1).to(device)
y_cheb = runge_function(x_cheb, beta=beta_val)

x_train, y_train = x_cheb, y_cheb

# ===== 修改 7: 测试集设计（实验.md：切比雪夫节点 + 密集采样）=====
# 测试集：切比雪夫节点 (100点) + 密集突变区采样 ([-0.2, 0.2]内300点)
i_test = torch.arange(100).float()
x_test_cheb = torch.cos(i_test * np.pi / 99).view(-1, 1)
x_test_dense = (torch.rand(300, 1) * 0.4) - 0.2  # [-0.2, 0.2] Uniform
x_test = torch.cat([x_test_cheb, x_test_dense], dim=0).to(device)
x_test, _ = torch.sort(x_test, dim=0)  # 排序以方便画图
y_test = runge_function(x_test, beta=beta_val)

# --- 阶段 1: 初始训练 (增加自动停止与 Test Loss 记录) ---
print(f"\n========== Stage 1: Initial Training (Chebyshev N={n_points}) ==========")
train_losses_s1 = []
test_losses_s1 = []
model_at_stop = None

for epoch in range(5001): # 设定一个较大的上限
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    train_losses_s1.append(loss.item())

    # 记录 Test Loss 用于过拟合判断
    model.eval()
    with torch.no_grad():
        t_loss = criterion(model(x_test), y_test).item()
        test_losses_s1.append(t_loss)

    # 自动停止检测: Loss < 1e-7
    if loss.item() < 1e-7:
        print(f">>> 目标 Loss 达到 {loss.item():.2e}，此时的Test loss达到{t_loss:.2e}停止训练 (Epoch {epoch})")
        model_at_stop = copy.deepcopy(model)
        plot_results(model, train_losses_s1, test_losses_s1, f"Stage1_Final_Stop_Epoch{epoch}",
                     x_train, y_train, x_test, y_test, beta=beta_val)
        break

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.2e}, Test Loss: {t_loss:.2e}")
        plot_results(model, train_losses_s1, test_losses_s1, f"Stage1_Training_{epoch}epochs",
                     x_train, y_train, x_test, y_test, beta=beta_val)

if model_at_stop is None: model_at_stop = copy.deepcopy(model)

# ===== 修改 8: 阶段 2 改为软剪枝准备（不执行硬剪枝）=====
model_ft = model_at_stop.to(device)
print("\n========== Stage 2: Soft Pruning Preparation ==========")
print(">>> 已激活软剪枝策略：通过加权L1正则直接抑制椭圆内奇点")
print(f"    目标椭圆参数: β={beta_val}")
plot_results(model_ft, [], [], "Stage2_Soft_Pruning_Init", x_train, y_train, x_test, y_test, beta=beta_val)

# ===== 修改 9: 阶段 3 改为软剪枝微调（实验.md：10000-20000 epochs，学习率衰减）=====
print("\n========== Stage 3: Fine-tuning with Soft Pruning ==========")
train_losses_s3 = []
test_losses_s3 = []
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)  # 初始学习率 1e-4
# ===== 修改 10: 添加学习率衰减（每 2000 epochs 衰减 50%）=====
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=2000, gamma=0.5)
lambda_reg = 1.0  # 软剪枝正则系数 (increased to 1.0)

for epoch in range(5001):  # 减少到 5000 epochs
    model_ft.train()
    optimizer_ft.zero_grad()
    outputs = model_ft(x_train)
    mse_loss = criterion(outputs, y_train)
    
    # ===== 修改 11: 计算软剪枝正则项并加入损失 =====
    # Warmup: 逐步增加正则项权重（前 3000 epochs）
    warmup_epochs = 3000
    current_lambda = lambda_reg * min(1.0, epoch / warmup_epochs)
    soft_penalty = compute_soft_pruning_penalty(model_ft, beta=beta_val)
    if epoch % 500 == 0:
        print(f"  Soft Penalty: {soft_penalty.item():.4e}")
    total_loss = mse_loss + current_lambda * soft_penalty
    
    total_loss.backward()
    optimizer_ft.step()
    scheduler.step()

    train_losses_s3.append(mse_loss.item())
    model_ft.eval()
    with torch.no_grad():
        test_loss_s3 = criterion(model_ft(x_test), y_test).item()
        test_losses_s3.append(test_loss_s3)

    if epoch % 500 == 0:
        print(f"Fine-tune Epoch {epoch}, Train Loss: {mse_loss.item():.2e}, "
              f"Test Loss: {test_loss_s3:.2e}, λ={current_lambda:.4f}")
        plot_results(model_ft, train_losses_s3, test_losses_s3, f"Stage3_FineTuning_{epoch}epochs",
                     x_train, y_train, x_test, y_test, beta=beta_val, prev_train_losses=train_losses_s1, prev_test_losses=test_losses_s1)

print(f"\n实验全部完成，结果保存在: {save_dir}")