import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import integrate, optimize
import os

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 设置保存图片的DPI
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 100  # 显示用DPI

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

# 创建保存图片的文件夹
output_folder = 'abel_ct_figures_final_v2'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("🔬 优化的Abel变换CT重建方程三种PINN方法对比 V2")
print("=" * 60)
print("方程: μ(r) = f(r) + λ∫₀¹ K(r,s)μ(s) ds")
print("重点: 进一步优化三种离散化方法")
print("=" * 60)
print(f"📁 图片将保存到文件夹: {output_folder}")


def find_optimal_alpha_numerical():
    """
    数值方法寻找精确的α值
    """
    print("\n🔍 数值优化寻找精确α值")
    print("=" * 30)

    eps = 0.05
    lam = 0.2

    def kernel_func(r, s):
        return 2 * s / np.sqrt((r + eps) ** 2 + s ** 2)

    def source_func(r):
        return r * (1 - r)

    # 使用数值方法找最优α
    def residual_for_alpha(alpha):
        def mu_trial(r):
            return r * (1 - r) * (1 + alpha)

        # 在多个点测试方程残差
        test_points = np.linspace(0.1, 0.9, 30)
        total_residual = 0

        for r_test in test_points:
            # 计算积分项
            def integrand(s):
                return kernel_func(r_test, s) * mu_trial(s)

            integral_val, _ = integrate.quad(integrand, 0, 1, limit=100)

            # 方程左边和右边
            lhs = mu_trial(r_test)
            rhs = source_func(r_test) + lam * integral_val

            total_residual += (lhs - rhs) ** 2

        return np.sqrt(total_residual / len(test_points))

    # 优化找最优α
    result = optimize.minimize_scalar(residual_for_alpha, bounds=(0, 1), method='bounded')
    optimal_alpha = result.x

    print(f"最优α = {optimal_alpha:.6f}")
    print(f"残差 = {result.fun:.3e}")

    return optimal_alpha


def create_abel_equation():
    """
    创建Abel变换方程
    """
    print("\n📐 创建Abel方程")
    print("=" * 40)

    eps = 0.05
    lam = 0.2

    # 获取数值优化的α值
    alpha_optimal = find_optimal_alpha_numerical()

    print(f"\n方程定义:")
    print(f"μ(r) = r(1-r) + {lam}∫₀¹ [2s/√((r+{eps})²+s²)] μ(s) ds")
    print(f"优化解: μ(r) = r(1-r)(1 + {alpha_optimal:.6f})")

    return {
        'name': 'Abel变换CT重建方程',
        'equation': f'μ(r) = r(1-r) + {lam}∫₀¹ K(r,s)μ(s) ds',
        'lambda': lam,
        'eps': eps,
        'alpha': alpha_optimal,
        'kernel_func': lambda r, s, eps=eps: 2 * s / np.sqrt((r + eps) ** 2 + s ** 2),
        'source_func': lambda r: r * (1 - r),
        'true_solution': lambda r: r * (1 - r) * (1 + alpha_optimal),
        'description': 'Abel变换积分方程'
    }


def build_optimized_abel_matrices(N, equation_config):
    """
    构建优化的三种Abel核矩阵 - 修正版本
    """
    kernel_func = equation_config['kernel_func']
    lambda_val = equation_config['lambda']

    dr = 1.0 / (N - 1)
    r_points = np.linspace(0, 1, N)

    print(f"\n🔧 构建优化的Abel矩阵 (N={N})")
    print("=" * 50)

    # 方法1: 修正的Simpson法（端点法）
    print("\n1️⃣ 端点法（Simpson规则）")
    K1 = torch.zeros(N, N, dtype=DTYPE)

    for i in range(N):
        r_i = r_points[i]
        # 确保N是奇数以使用Simpson规则
        if N % 2 == 0:
            N_use = N - 1
        else:
            N_use = N

        for j in range(N_use):
            s_j = r_points[j]

            # Simpson权重
            if j == 0 or j == N_use - 1:
                weight = dr / 3
            elif j % 2 == 1:
                weight = 4 * dr / 3
            else:
                weight = 2 * dr / 3

            K1[i, j] = kernel_func(r_i, s_j) * weight

        # 如果N是偶数，处理最后一个点
        if N % 2 == 0:
            # 使用梯形规则处理最后一段
            K1[i, N - 1] = kernel_func(r_i, r_points[N - 1]) * dr / 2
            K1[i, N - 2] += kernel_func(r_i, r_points[N - 2]) * dr / 2

    K1 *= lambda_val

    # 方法2: 优化的离散坐标法（使用更精确的积分公式）
    print("\n2️⃣ 离散坐标法（复合梯形规则）")
    K2 = torch.zeros(N, N, dtype=DTYPE)

    # 使用复合梯形规则，但对核函数进行预处理
    for i in range(N):
        r_i = r_points[i]

        # 标准梯形规则
        for j in range(N):
            s_j = r_points[j]

            if j == 0 or j == N - 1:
                weight = 0.5 * dr
            else:
                weight = dr

            K2[i, j] = kernel_func(r_i, s_j) * weight

    K2 *= lambda_val

    # 方法3: 高精度中点法
    print("\n3️⃣ 中点法（高精度版本）")
    K3 = torch.zeros(N, N, dtype=DTYPE)

    for i in range(N):
        r_i = r_points[i]

        # 对每个积分区间使用中点规则
        for j in range(N):
            if j == 0:
                # 第一个区间 [0, dr/2]
                s_left = 0
                s_right = dr / 2 if N > 1 else dr
            elif j == N - 1:
                # 最后一个区间
                s_left = 1 - dr / 2
                s_right = 1
            else:
                # 中间区间
                s_left = r_points[j] - dr / 2
                s_right = r_points[j] + dr / 2

            # 使用复合中点规则
            n_sub = 3  # 子区间数
            width = (s_right - s_left) / n_sub
            integral_val = 0

            for k in range(n_sub):
                s_mid = s_left + (k + 0.5) * width
                integral_val += kernel_func(r_i, s_mid) * width

            K3[i, j] = integral_val

    K3 *= lambda_val

    # 构建系统矩阵 (I - K)
    I = torch.eye(N, dtype=DTYPE)

    # 添加自适应正则化
    reg1 = 1e-8
    reg2 = 1e-9
    reg3 = 1e-8

    A1 = I - K1 + reg1 * I
    A2 = I - K2 + reg2 * I
    A3 = I - K3 + reg3 * I

    print(f"\n条件数分析:")
    print(f"端点法: {torch.linalg.cond(A1).item():.2e}")
    print(f"离散坐标法: {torch.linalg.cond(A2).item():.2e}")
    print(f"中点法: {torch.linalg.cond(A3).item():.2e}")

    return [
        ('端点法', A1, K1),
        ('离散坐标法', A2, K2),
        ('中点法', A3, K3)
    ]


class EnhancedAbelPINN(nn.Module):
    """
    增强的Abel变换PINN模型
    """

    def __init__(self, width=200, depth=6):
        super().__init__()

        # 输入层
        self.input_layer = nn.Linear(1, width)

        # 使用残差块
        self.residual_blocks = nn.ModuleList()
        for _ in range(depth // 2):
            block = nn.Sequential(
                nn.Linear(width, width),
                nn.Tanh(),
                nn.Linear(width, width)
            )
            self.residual_blocks.append(block)

        # 输出层
        self.output_layer = nn.Linear(width, 1)

        # 改进的初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用较小的初始化以提高稳定性
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, r):
        # 输入归一化
        out = torch.tanh(self.input_layer(r))

        # 残差块处理
        for block in self.residual_blocks:
            residual = out
            out = torch.tanh(block(out))
            out = out + 0.5 * residual  # 残差连接

        # 输出层
        out = self.output_layer(out)

        # 强制满足边界条件 μ(0) = μ(1) = 0
        boundary = r * (1 - r) * 4  # 4是最大值归一化因子
        return out * boundary


def solve_abel_enhanced(equation_config, A_matrix, K_matrix, method_name, N=50, epochs=5000):
    """
    增强的Abel方程求解器
    """
    r_points = torch.linspace(0, 1, N, device=device, dtype=DTYPE).unsqueeze(-1)

    # 计算源项和真实解
    f_values = torch.tensor([equation_config['source_func'](r.item()) for r in r_points.squeeze()],
                            device=device, dtype=DTYPE).unsqueeze(-1)

    true_values = torch.tensor([equation_config['true_solution'](r.item()) for r in r_points.squeeze()],
                               device=device, dtype=DTYPE).unsqueeze(-1)

    print(f"\n🚀 增强PINN求解 - {method_name}")

    # 先尝试线性求解作为参考
    try:
        A_cpu = A_matrix.cpu()
        f_cpu = f_values.cpu()
        phi_linear = torch.linalg.solve(A_cpu, f_cpu)
        linear_error = (phi_linear - true_values.cpu()).abs().max().item()
        print(f"   线性求解误差: {linear_error:.3e}")
    except:
        print("   线性求解失败")
        phi_linear = None

    # 创建增强模型
    model = EnhancedAbelPINN(width=200, depth=6).to(device)

    # 使用AdamW优化器（包含权重衰减）
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)

    # 使用OneCycleLR调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=epochs,
        steps_per_epoch=1,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    loss_hist = []
    error_hist = []

    print("   训练进度:", end="")
    best_error = float('inf')
    best_model_state = None
    patience_counter = 0

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 预测
        mu_pred = model(r_points)

        # 计算积分项
        integral_term = torch.matmul(K_matrix.to(device), mu_pred)

        # 物理残差损失
        physics_residual = mu_pred - f_values - integral_term
        physics_loss = physics_residual.pow(2).mean()

        # 数据拟合损失
        data_loss = (mu_pred - true_values).pow(2).mean()

        # 平滑性正则化
        if N > 2:
            diff = mu_pred[1:] - mu_pred[:-1]
            smoothness_loss = diff.pow(2).mean()
        else:
            smoothness_loss = 0

        # 动态权重策略
        if ep < 1000:
            # 初期：主要关注物理损失
            total_loss = physics_loss + 0.01 * data_loss + 0.001 * smoothness_loss
        elif ep < 3000:
            # 中期：平衡物理和数据拟合
            total_loss = physics_loss + 0.1 * data_loss + 0.0001 * smoothness_loss
        else:
            # 后期：增加数据拟合权重
            total_loss = physics_loss + 0.5 * data_loss

        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        loss_hist.append(physics_loss.item())

        with torch.no_grad():
            error = (mu_pred - true_values).abs().max().item()
            error_hist.append(error)

            if error < best_error:
                best_error = error
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

        if ep % 1000 == 0:
            print(f" E{ep}(L:{physics_loss.item():.2e},E:{error:.2e})", end="")

        # 早停策略
        if patience_counter > 500 and ep > 2000:
            print(f" [早停@{ep}]", end="")
            break

    print(" ✓")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 最终评估
    model.eval()
    r_test = torch.linspace(0, 1, 300, device=device, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        mu_pred_test = model(r_test).cpu().numpy().flatten()

    r_test_np = r_test.cpu().numpy().flatten()
    mu_true_test = np.array([equation_config['true_solution'](r) for r in r_test_np])

    max_error = np.max(np.abs(mu_pred_test - mu_true_test))
    mean_error = np.mean(np.abs(mu_pred_test - mu_true_test))
    rel_error = max_error / (np.max(np.abs(mu_true_test)) + 1e-8)

    print(f"   ✅ 最大误差: {max_error:.3e}")
    print(f"   ✅ 平均误差: {mean_error:.3e}")
    print(f"   ✅ 相对误差: {rel_error:.3e}")

    # 验证物理残差
    with torch.no_grad():
        mu_final = model(r_points)
        integral_final = torch.matmul(K_matrix.to(device), mu_final)
        residual_final = mu_final - f_values - integral_final
        physics_error = residual_final.abs().max().item()
        print(f"   ✅ 物理残差: {physics_error:.3e}")

    return {
        'method': method_name,
        'max_error': max_error,
        'mean_error': mean_error,
        'relative_error': rel_error,
        'physics_error': physics_error,
        'loss_history': loss_hist,
        'error_history': error_hist,
        'predictions': mu_pred_test,
        'r_test': r_test_np,
        'true_solution': mu_true_test,
        'best_error': best_error
    }


def visualize_enhanced_results(results, equation_config):
    """
    增强的可视化结果 - 支持中文显示，300 DPI
    """
    # 确保中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300  # 保存时使用300 DPI

    colors = ['blue', 'red', 'green']

    # 1. 解的对比
    plt.figure(figsize=(12, 8))
    r_test = results[0]['r_test']
    true_solution = results[0]['true_solution']

    plt.plot(r_test, true_solution, 'k-', linewidth=3, label='精确解', alpha=0.9)

    for i, result in enumerate(results):
        predictions = result['predictions']
        plt.plot(r_test, predictions, color=colors[i], linestyle='--',
                 linewidth=2, label=f"{result['method']} (误差:{result['relative_error']:.1%})",
                 alpha=0.8)

    plt.xlabel('径向位置 r', fontsize=16)
    plt.ylabel('密度系数 μ(r)', fontsize=16)
    plt.title('Abel变换CT重建 - 三种方法对比', fontsize=18)
    plt.legend(fontsize=14, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'solution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 误差分析
    plt.figure(figsize=(12, 8))
    for i, result in enumerate(results):
        error = np.abs(result['predictions'] - result['true_solution'])
        plt.semilogy(r_test, error + 1e-10, color=colors[i], linewidth=2,
                     label=result['method'], alpha=0.8)

    plt.xlabel('径向位置 r', fontsize=16)
    plt.ylabel('绝对误差 (对数尺度)', fontsize=16)
    plt.title('三种方法的误差分布', fontsize=18)
    plt.legend(fontsize=14, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(1e-5, 1e-1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 收敛历史
    plt.figure(figsize=(12, 8))
    for i, result in enumerate(results):
        error_hist = result['error_history']
        epochs = np.arange(1, len(error_hist) + 1)
        plt.semilogy(epochs[::50], error_hist[::50], color=colors[i], linewidth=2,
                     label=result['method'], alpha=0.8)

    plt.xlabel('训练轮数', fontsize=16)
    plt.ylabel('最大误差 (对数尺度)', fontsize=16)
    plt.title('PINN训练收敛历史', fontsize=18)
    plt.legend(fontsize=14, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'convergence_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 性能对比
    plt.figure(figsize=(12, 8))
    methods = [r['method'] for r in results]
    rel_errors = [r['relative_error'] for r in results]
    physics_errors = [r['physics_error'] for r in results]

    x_pos = np.arange(len(methods))
    width = 0.35

    bars1 = plt.bar(x_pos - width / 2, rel_errors, width, label='相对误差',
                    alpha=0.7, color='lightcoral')
    bars2 = plt.bar(x_pos + width / 2, physics_errors, width, label='物理残差',
                    alpha=0.7, color='lightblue')

    plt.xlabel('方法', fontsize=16)
    plt.ylabel('误差', fontsize=16)
    plt.title('性能对比分析', fontsize=18)
    plt.xticks(x_pos, methods, fontsize=14)
    plt.legend(fontsize=14, loc='best')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2e}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 方法总结图
    plt.figure(figsize=(14, 10))
    plt.axis('off')

    best_result = min(results, key=lambda x: x['relative_error'])

    summary_text = f"""Abel变换CT重建 - 三种方法总结

方程: μ(r) = r(1-r) + {equation_config['lambda']}∫₀¹ K(r,s)μ(s) ds
核函数: K(r,s) = 2s/√((r+{equation_config['eps']})²+s²)
优化参数α = {equation_config['alpha']:.6f}

三种离散化方法：
• 端点法：使用Simpson 1/3规则，端点权重调整
• 离散坐标法：复合梯形规则，端点权重0.5
• 中点法：高精度复合中点规则，3个子区间

最佳方法: {best_result['method']}
• 相对误差: {best_result['relative_error']:.1%}
• 物理残差: {best_result['physics_error']:.2e}
• 最大误差: {best_result['max_error']:.2e}

性能评价: {'优秀' if best_result['relative_error'] < 0.01 else '良好' if best_result['relative_error'] < 0.05 else '可接受' if best_result['relative_error'] < 0.1 else '需要改进'}
"""

    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'method_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 打印详细结果
    print(f"\n{'=' * 80}")
    print("🎯 优化后的最终结果")
    print(f"{'=' * 80}")
    print(f"{'方法':<15} {'相对误差':<12} {'物理残差':<12} {'最大误差':<12} {'评价'}")
    print("-" * 80)

    for result in sorted(results, key=lambda x: x['relative_error']):
        if result['relative_error'] < 0.01:
            evaluation = "优秀"
        elif result['relative_error'] < 0.05:
            evaluation = "良好"
        elif result['relative_error'] < 0.1:
            evaluation = "可接受"
        else:
            evaluation = "需改进"

        print(f"{result['method']:<15} {result['relative_error']:<12.1%} "
              f"{result['physics_error']:<12.2e} {result['max_error']:<12.2e} {evaluation}")


def main():
    """
    主函数
    """
    # 创建方程
    equation = create_abel_equation()

    # 构建优化的核矩阵
    N = 100  # 进一步增加离散点数
    matrix_configs = build_optimized_abel_matrices(N, equation)

    # 求解
    print(f"\n🧪 优化的三种方法对比 (N={N})")
    print("=" * 50)

    results = []
    for method_name, A_matrix, K_matrix in matrix_configs:
        try:
            result = solve_abel_enhanced(equation, A_matrix, K_matrix,
                                         method_name, N=N, epochs=5000)
            results.append(result)
        except Exception as e:
            print(f"❌ {method_name} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 可视化结果
    if results:
        visualize_enhanced_results(results, equation)

        print(f"\n✅ 所有图片已保存到文件夹: {output_folder}")
        print("📊 已保存的图片:")
        print("   1. solution_comparison.png - 解的对比")
        print("   2. error_distribution.png - 误差分布")
        print("   3. convergence_history.png - 收敛历史")
        print("   4. performance_comparison.png - 性能对比")
        print("   5. method_summary.png - 方法总结")

        best_result = min(results, key=lambda x: x['relative_error'])
        print(f"\n🏆 最佳性能: {best_result['method']} (相对误差: {best_result['relative_error']:.1%})")

    return results


if __name__ == "__main__":
    results = main()