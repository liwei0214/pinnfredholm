import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

# 设置matplotlib中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei',
                                   'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建保存图片的文件夹（避免与奇异核代码冲突）
output_folder = 'normal_kernel_figures_optimized'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("🔍 正常核Fredholm积分方程三种PINN方法对比")
print("=" * 60)
print("方程: φ(s) + ∫₀¹ se^t φ(t) dt = e^(-s)")
print("精确解: φ(s) = e^(-s) - s/2")
print("=" * 60)
print(f"📁 图片将保存到文件夹: {output_folder}")


def create_normal_kernel_equation():
    """
    创建正常核Fredholm积分方程
    φ(s) + ∫₀¹ se^t φ(t) dt = e^(-s)
    重写为: φ(s) = e^(-s) - ∫₀¹ se^t φ(t) dt
    """
    print("\n📐 Step 1: 数学验证")
    print("=" * 40)
    print("原方程: φ(s) + ∫₀¹ se^t φ(t) dt = e^(-s)")
    print("标准形式: φ(s) = f(s) + λ∫₀¹ K(s,t)φ(t) dt")
    print("其中:")
    print("  f(s) = e^(-s)")
    print("  K(s,t) = se^t")
    print("  λ = -1")
    print("  精确解: φ(s) = e^(-s) - s/2")

    # 验证精确解
    def verify_solution():
        s_test = 0.5
        phi_exact = np.exp(-s_test) - s_test / 2

        # 计算积分项 ∫₀¹ se^t φ(t) dt，其中 φ(t) = e^(-t) - t/2
        def integrand(t):
            return s_test * np.exp(t) * (np.exp(-t) - t / 2)

        # 解析积分: ∫₀¹ s*e^t*(e^(-t) - t/2) dt = ∫₀¹ s*(1 - t*e^t/2) dt
        # = s*[t - (t*e^t - e^t)/2]₀¹ = s*[1 - (e - e)/2 + e/2] = s*(1 + e/2 - 1/2)
        integral_exact = s_test * (1 - 0.5 + (np.e - 1) / 2)

        lhs = phi_exact + integral_exact
        rhs = np.exp(-s_test)

        print(f"\n验证 s={s_test}:")
        print(f"φ({s_test}) = {phi_exact:.6f}")
        print(f"积分项 = {integral_exact:.6f}")
        print(f"左边 = {lhs:.6f}")
        print(f"右边 = {rhs:.6f}")
        print(f"误差 = {abs(lhs - rhs):.2e}")

    verify_solution()

    return {
        'name': '正常核Fredholm积分方程',
        'lambda': -1.0,
        'kernel_func': lambda s, t: s * np.exp(t),
        'source_func': lambda s: np.exp(-s),
        'true_solution': lambda s: np.exp(-s) - s / 2,
        'description': 'φ(s) + ∫₀¹ se^t φ(t) dt = e^(-s)'
    }


def build_kernel_matrices_normal(N, kernel_func, lambda_val):
    """
    构建三种离散化方法的核矩阵（正常核）
    """
    print(f"\n🔬 Step 2: 三种方法核矩阵构建 (N={N})")
    print("=" * 50)

    dx = 1.0 / (N - 1)
    s_points = np.linspace(0, 1, N)

    # 方法1: 端点法（梯形积分）
    print("\n1️⃣ 端点法（梯形积分）")
    K1 = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            K1[i, j] = kernel_func(s_points[i], s_points[j]) * dx

    # 梯形规则：边界权重×0.5
    K1[:, 0] *= 0.5
    K1[:, -1] *= 0.5
    K1 *= lambda_val
    print(f"K1[{N // 2},:5] = {K1[N // 2, :5]}")

    # 方法2: 离散坐标法（矩形积分）
    print("\n2️⃣ 离散坐标法（矩形积分）")
    K2 = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            K2[i, j] = kernel_func(s_points[i], s_points[j]) * dx

    K2 *= lambda_val
    print(f"K2[{N // 2},:5] = {K2[N // 2, :5]}")

    # 方法3: 中点法
    print("\n3️⃣ 中点法")
    K3 = torch.zeros(N, N)
    dx_mid = 1.0 / N

    for i in range(N):
        s_i = s_points[i]
        for j in range(N):
            if j < N - 1:
                t_mid = (j + 0.5) * dx_mid
                K3[i, j] = kernel_func(s_i, t_mid) * dx_mid
            else:
                t_end = min(1.0, (j + 0.5) * dx_mid)
                K3[i, j] = kernel_func(s_i, t_end) * dx_mid * 0.5

    K3 *= lambda_val
    print(f"K3[{N // 2},:5] = {K3[N // 2, :5]}")

    # 分析条件数
    I = torch.eye(N, dtype=DTYPE)
    A1 = I - K1
    A2 = I - K2
    A3 = I - K3

    cond1 = torch.linalg.cond(A1).item()
    cond2 = torch.linalg.cond(A2).item()
    cond3 = torch.linalg.cond(A3).item()

    print(f"\n🎯 矩阵条件数分析:")
    print(f"端点法条件数:     {cond1:.2e}")
    print(f"离散坐标法条件数: {cond2:.2e}")
    print(f"中点法条件数:     {cond3:.2e}")

    return K1, K2, K3


class NormalKernelPINN(nn.Module):
    """
    正常核PINN模型
    """

    def __init__(self, width=128, depth=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(1, width))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))

        self.net = nn.Sequential(*layers)

        # 初始化
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s):
        return self.net(s)


def solve_normal_kernel(equation, K_matrix, method_name, N=50, epochs=3000):
    """
    求解正常核积分方程
    """
    s_points = torch.linspace(0, 1, N, device=device, dtype=DTYPE).unsqueeze(-1)

    # 计算源项
    f_values = torch.tensor([equation['source_func'](s.item()) for s in s_points.squeeze()],
                            device=device, dtype=DTYPE).unsqueeze(-1)

    # 真实解
    true_values = torch.tensor([equation['true_solution'](s.item()) for s in s_points.squeeze()],
                               device=device, dtype=DTYPE).unsqueeze(-1)

    print(f"\n🚀 PINN求解 - {method_name}")
    print(f"   核矩阵条件数: {torch.linalg.cond(K_matrix).item():.2e}")

    # 验证方程残差
    integral_term = torch.matmul(K_matrix.to(device), true_values)
    residual_check = true_values - f_values - integral_term
    print(f"   精确解残差: {residual_check.abs().max().item():.3e}")

    model = NormalKernelPINN(width=128, depth=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    loss_hist = []
    error_hist = []

    print("   训练进度:", end="")
    best_error = float('inf')

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        phi_pred = model(s_points)
        integral_term = torch.matmul(K_matrix.to(device), phi_pred)
        residual = phi_pred - f_values - integral_term
        loss = residual.pow(2).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_hist.append(loss.item())

        with torch.no_grad():
            error = (phi_pred - true_values).abs().max().item()
            error_hist.append(error)
            if error < best_error:
                best_error = error

        if ep % 600 == 0:
            print(f" E{ep}(L:{loss.item():.2e},E:{error:.2e})", end="")

    print(" ✓")

    # 最终评估
    model.eval()
    s_test = torch.linspace(0, 1, 200, device=device, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        phi_pred_test = model(s_test).cpu().numpy().flatten()

    s_test_np = s_test.cpu().numpy().flatten()
    phi_true_test = np.array([equation['true_solution'](s) for s in s_test_np])

    max_error = np.max(np.abs(phi_pred_test - phi_true_test))
    mean_error = np.mean(np.abs(phi_pred_test - phi_true_test))
    relative_error = max_error / (np.max(np.abs(phi_true_test)) + 1e-8)

    print(f"   ✅ 最大误差: {max_error:.3e}")
    print(f"   ✅ 平均误差: {mean_error:.3e}")
    print(f"   ✅ 相对误差: {relative_error:.3e}")

    return {
        'method': method_name,
        'max_error': max_error,
        'mean_error': mean_error,
        'relative_error': relative_error,
        'loss_history': loss_hist,
        'error_history': error_hist,
        'predictions': phi_pred_test,
        's_test': s_test_np,
        'true_solution': phi_true_test,
        'best_error': best_error
    }


def visualize_normal_kernel_results(results, equation):
    """
    优化的可视化 - 每个图单独保存，支持中文显示
    """
    # 重新设置matplotlib参数，确保中文显示
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['figure.dpi'] = 120  # 显示dpi
    plt.rcParams['savefig.dpi'] = 300  # 保存dpi
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei',
                                       'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']

    # 1. 解的对比
    plt.figure(figsize=(12, 8))
    s_test = results[0]['s_test']
    true_solution = results[0]['true_solution']

    plt.plot(s_test, true_solution, 'k-', linewidth=3, label='精确解: φ(s) = e^(-s) - s/2', alpha=0.9)

    for i, result in enumerate(results):
        predictions = result['predictions']
        plt.plot(s_test[::5], predictions[::5], markers[i], color=colors[i],
                 markersize=5, label=f"{result['method']} (相对误差:{result['relative_error']:.1e})",
                 alpha=0.8)

    plt.xlabel('s', fontsize=15)
    plt.ylabel('φ(s)', fontsize=15)
    plt.title('正常核Fredholm积分方程解的对比', fontsize=17, pad=20)
    plt.legend(fontsize=13, loc='best')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'solution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 误差分布
    plt.figure(figsize=(12, 8))
    for i, result in enumerate(results):
        error = np.abs(result['predictions'] - result['true_solution']) + 1e-15
        plt.semilogy(s_test, error, color=colors[i], linewidth=2.5,
                     label=f"{result['method']}", alpha=0.8)

    plt.xlabel('s', fontsize=15)
    plt.ylabel('绝对误差 (对数尺度)', fontsize=15)
    plt.title('三种方法的误差分布对比', fontsize=17, pad=20)
    plt.legend(fontsize=13, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 训练收敛
    plt.figure(figsize=(12, 8))
    for i, result in enumerate(results):
        loss_hist = result['loss_history']
        epochs = np.arange(1, len(loss_hist) + 1)
        plt.semilogy(epochs[::10], loss_hist[::10], color=colors[i], linewidth=2.5,
                     label=f"{result['method']}", alpha=0.8)

    plt.xlabel('训练轮数', fontsize=15)
    plt.ylabel('损失函数 (对数尺度)', fontsize=15)
    plt.title('PINN训练收敛历史对比', fontsize=17, pad=20)
    plt.legend(fontsize=13, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 误差收敛历史
    plt.figure(figsize=(12, 8))
    for i, result in enumerate(results):
        error_hist = result['error_history']
        epochs = np.arange(1, len(error_hist) + 1)
        plt.semilogy(epochs[::10], error_hist[::10], color=colors[i], linewidth=2.5,
                     label=f"{result['method']}", alpha=0.8)

    plt.xlabel('训练轮数', fontsize=15)
    plt.ylabel('最大误差 (对数尺度)', fontsize=15)
    plt.title('误差收敛历史对比', fontsize=17, pad=20)
    plt.legend(fontsize=13, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'error_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 核函数可视化
    plt.figure(figsize=(12, 8))
    s_vals = np.linspace(0, 1, 100)
    t_vals = np.linspace(0, 1, 100)
    S, T = np.meshgrid(s_vals, t_vals)
    K_vals = S * np.exp(T)

    contour = plt.contourf(S, T, K_vals, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, label='K(s,t) = se^t')
    plt.xlabel('s', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.title('正常核函数可视化: K(s,t) = se^t', fontsize=17, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'kernel_function.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 性能对比柱状图
    plt.figure(figsize=(12, 8))
    methods = [r['method'] for r in results]
    max_errors = [r['max_error'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    relative_errors = [r['relative_error'] for r in results]

    x_pos = np.arange(len(methods))
    width = 0.25

    plt.bar(x_pos - width, max_errors, width, label='最大误差', alpha=0.7, color='lightcoral')
    plt.bar(x_pos, mean_errors, width, label='平均误差', alpha=0.7, color='lightblue')
    plt.bar(x_pos + width, relative_errors, width, label='相对误差', alpha=0.7, color='lightgreen')

    plt.xlabel('方法', fontsize=15)
    plt.ylabel('误差值', fontsize=15)
    plt.title('三种方法的误差统计对比', fontsize=17, pad=20)
    plt.xticks(x_pos, methods, fontsize=12)
    plt.legend(fontsize=13)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'error_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 7. 精度排名柱状图
    plt.figure(figsize=(12, 8))
    sorted_results = sorted(results, key=lambda x: x['relative_error'])
    methods_sorted = [r['method'] for r in sorted_results]
    errors_sorted = [r['relative_error'] for r in sorted_results]

    bars = plt.bar(methods_sorted, errors_sorted, color=colors, alpha=0.7, width=0.6)
    plt.ylabel('相对误差', fontsize=15)
    plt.title('方法精度排名（按相对误差升序）', fontsize=17, pad=20)
    plt.yscale('log')

    # 添加数值标签
    for bar, err in zip(bars, errors_sorted):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height * 1.1,
                 f'{err:.2e}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'performance_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 8. 方法总结（作为文本图片）
    plt.figure(figsize=(14, 11))
    plt.axis('off')

    summary_text = f"""🔬 正常核Fredholm积分方程PINN求解结果总结

📐 方程信息：
   φ(s) + ∫₀¹ se^t φ(t) dt = e^(-s)
   标准形式: φ(s) = f(s) + λ∫₀¹ K(s,t)φ(t) dt
   其中: f(s) = e^(-s), K(s,t) = se^t, λ = -1
   精确解: φ(s) = e^(-s) - s/2

🏆 精度排名（按相对误差升序）：
"""

    for i, result in enumerate(sorted_results, 1):
        summary_text += f"\n{i}. {result['method']}"
        summary_text += f"\n   📊 相对误差: {result['relative_error']:.3e}"
        summary_text += f"\n   📈 最大误差: {result['max_error']:.3e}"
        summary_text += f"\n   📋 平均误差: {result['mean_error']:.3e}"
        summary_text += f"\n   🎯 最佳误差: {result['best_error']:.3e}\n"

    summary_text += f"""
🔧 三种数值方法特点：
• ✅ 端点法（梯形积分）: 经典数值积分，边界权重减半
• ✅ 离散坐标法（矩形积分）: 直接离散化，计算简单
• ✅ 中点法: 使用中点规则，提高积分精度

📈 技术亮点：
• 🎯 正常核函数无奇异性，数值稳定
• 🎯 三种离散化方法全面对比
• 🎯 PINN有效求解积分方程
• 🎯 高精度求解(最优相对误差: {sorted_results[0]['relative_error']:.2e})

🎨 可视化特色：
• 解的对比曲线 • 误差分布分析 • 训练收敛历史
• 核函数等高线图 • 性能统计对比 • 精度排名展示
"""

    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=13,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightcyan", alpha=0.8, edgecolor='teal'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'method_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 打印详细结果
    print(f"\n{'=' * 80}")
    print("🏆 正常核方程结果总结")
    print(f"{'=' * 80}")
    print(f"{'方法':<15} {'相对误差':<15} {'最大误差':<15} {'平均误差':<15}")
    print("-" * 80)

    for result in sorted_results:
        print(f"{result['method']:<15} {result['relative_error']:<15.3e} "
              f"{result['max_error']:<15.3e} {result['mean_error']:<15.3e}")

    print(f"\n✅ 所有图片已保存到文件夹: {output_folder}")
    print("📊 已保存的高清图片 (300 DPI):")
    print("   1. solution_comparison.png - 解的对比")
    print("   2. error_distribution.png - 误差分布")
    print("   3. training_convergence.png - 训练收敛")
    print("   4. error_convergence.png - 误差收敛历史")
    print("   5. kernel_function.png - 核函数可视化")
    print("   6. error_statistics.png - 误差统计")
    print("   7. performance_ranking.png - 精度排名")
    print("   8. method_summary.png - 方法总结")


def main_normal_kernel():
    """
    正常核方程主函数
    """
    # 创建方程
    equation = create_normal_kernel_equation()

    # 构建核矩阵
    N = 50
    K1, K2, K3 = build_kernel_matrices_normal(N, equation['kernel_func'], equation['lambda'])

    # 求解
    print(f"\n🧪 Step 3: 三种方法求解对比 (N={N})")
    print("=" * 40)

    methods = [
        ('端点法', K1),
        ('离散坐标法', K2),
        ('中点法', K3)
    ]

    results = []
    for method_name, K_matrix in methods:
        try:
            result = solve_normal_kernel(equation, K_matrix, method_name, N=N, epochs=2400)
            results.append(result)
        except Exception as e:
            print(f"❌ {method_name} 失败: {e}")

    # 可视化
    if results:
        visualize_normal_kernel_results(results, equation)

    return results


if __name__ == "__main__":
    results = main_normal_kernel()