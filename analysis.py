# -*- coding: utf-8 -*-
import math
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------------------------------------------------------
# 0. 原始数据（一周分流测试）
# ----------------------------------------------------------------------
n_A, x_A = 1200, 72      # 对照组：访客数、购买数
n_B, x_B = 1180, 95      # 实验组：访客数、购买数
alpha = 0.05             # 预先设定的单侧显著性水平

# ----------------------------------------------------------------------
# 1. 原始转化率
# ----------------------------------------------------------------------
p_A, p_B = x_A / n_A, x_B / n_B
delta = p_B - p_A
lift = delta / p_A

# ----------------------------------------------------------------------
# 2. 双比例 z 检验（合并方差，单侧备择 p_B > p_A）
# ----------------------------------------------------------------------
p_pool = (x_A + x_B) / (n_A + n_B)
se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_A + 1 / n_B))
z_stat = delta / se_pool
p_one_sided = stats.norm.sf(z_stat)
p_two_sided = 2 * p_one_sided

# 2x2 列联表 Pearson 卡方检验（不作 Yates 校正，df = 1）
table = np.array([[x_A, n_A - x_A], [x_B, n_B - x_B]])
chi2, p_chi2, dof, expected = stats.chi2_contingency(table, correction=False)

# ----------------------------------------------------------------------
# 3. 效应量置信区间（非合并方差）
# ----------------------------------------------------------------------
se_unpooled = math.sqrt(p_A * (1 - p_A) / n_A + p_B * (1 - p_B) / n_B)
z_975 = stats.norm.ppf(0.975)
z_95 = stats.norm.ppf(0.95)
ci_low_2s, ci_high_2s = delta - z_975 * se_unpooled, delta + z_975 * se_unpooled
ci_low_1s = delta - z_95 * se_unpooled

# 每组转化率的 Wilson 95% 置信区间（用于绘图误差线）
def wilson(x, n, z=z_975):
    p = x / n
    den = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return center - half, center + half

ci_A, ci_B = wilson(x_A, n_A), wilson(x_B, n_B)

# ----------------------------------------------------------------------
# 4. 统计功效与最小样本量
# ----------------------------------------------------------------------
power_obs = stats.norm.cdf(delta / se_pool - z_95)

def min_sample_size(p1, p2, a=0.05, power=0.80):
    za, zb = stats.norm.ppf(1 - a), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    term1 = za * math.sqrt(2 * pbar * (1 - pbar))
    term2 = zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((term1 + term2) ** 2 / (p2 - p1) ** 2)

n_min = min_sample_size(p_A, p_B)

def power_at(n_per_group):
    se = math.sqrt(p_pool * (1 - p_pool) * (2 / n_per_group))
    return stats.norm.cdf(delta / se - z_95)

# ----------------------------------------------------------------------
# 5. 分流比例失衡（SRM）健康检查
# ----------------------------------------------------------------------
exp_n = (n_A + n_B) / 2
chi2_srm = (n_A - exp_n) ** 2 / exp_n + (n_B - exp_n) ** 2 / exp_n
p_srm = stats.chi2.sf(chi2_srm, 1)

# ----------------------------------------------------------------------
# 6. 输出测试运行结果
# ----------------------------------------------------------------------
print("=" * 62)
print("DSCI 6000 作业1 | A/B 测试统计输出（单侧 alpha = 0.05）")
print("=" * 62)
print(f"[1] 原始转化率      A = {x_A}/{n_A} = {p_A*100:.4f}%   "
      f"B = {x_B}/{n_B} = {p_B*100:.4f}%")
print(f"    绝对提升        {delta*100:.4f} pp    相对提升 {lift*100:.2f}%")
print(f"[2] 合并比例        p_pool = {p_pool:.6f}   SE_pool = {se_pool:.6f}")
print(f"    双比例 z 检验   z = {z_stat:.4f}   p(单侧) = {p_one_sided:.4f}   "
      f"p(双侧) = {p_two_sided:.4f}")
print(f"    Pearson 卡方    chi2(1) = {chi2:.4f}   p = {p_chi2:.4f}   "
      f"(chi2 = z^2 = {z_stat**2:.4f})")
print(f"[3] 95% CI(双侧)   [{ci_low_2s*100:.4f} pp, {ci_high_2s*100:.4f} pp]")
print(f"    95% CI(单侧)   下限 = {ci_low_1s*100:.4f} pp")
print(f"[4] 当前功效        power = {power_obs*100:.2f}%  (n≈1200/组)")
print(f"    最小样本量      每组 {n_min} 人，合计 {2*n_min} 人；"
      f"该样本量下 power = {power_at(n_min)*100:.2f}%")
print("    MDE-样本量      1.0pp -> 7514/组 | 1.5pp -> 3459/组 | "
      "2.0pp -> 2012/组 | 2.05pp -> 1920/组")
print(f"[5] SRM 健康检查    chi2(1) = {chi2_srm:.4f}   p = {p_srm:.4f} "
      f"(A 占比 {n_A/(n_A+n_B)*100:.2f}%)")
print("[6] 统计决策        p(单侧) < 0.05 -> 拒绝 H0；"
      "但双侧 CI 含 0，证据强度为临界弱阳性")
print("=" * 62)

# ----------------------------------------------------------------------
# 7. 图 1：转化率对比条形图（含 95% CI 与测试信息标注）
# ----------------------------------------------------------------------
candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]
available = {f.name for f in font_manager.fontManager.ttflist}
cn_font = next((f for f in candidates if f in available), None)
if cn_font:
    plt.rcParams["font.sans-serif"] = [cn_font]
else:
    cn_font = "DejaVu Sans (Latin labels)"
plt.rcParams["axes.unicode_minus"] = False
print(f"[7] 图表字体        {cn_font}")

use_cn = cn_font not in ("DejaVu Sans (Latin labels)",)
labels = ["对照组 A\n原版 UI 布局", "实验组 B\n改版 UI 布局"] if use_cn else ["Control A\nOriginal UI", "Variant B\nRedesigned UI"]
title = "图 1  A/B 组购买转化率对比（误差线为各组 95% Wilson 置信区间）" if use_cn \
        else "Figure 1  Purchase conversion rate by group (95% Wilson CI)"
ylabel = "购买转化率（%）" if use_cn else "Purchase conversion rate (%)"

fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
rates = [p_A * 100, p_B * 100]
err_low = [p_A * 100 - ci_A[0] * 100, p_B * 100 - ci_B[0] * 100]
err_high = [ci_A[1] * 100 - p_A * 100, ci_B[1] * 100 - p_B * 100]
bars = ax.bar(labels, rates, width=0.46, color=["#7CA6D8", "#E8834A"],
              edgecolor="#333333", linewidth=0.8, zorder=3)
ax.errorbar([0, 1], rates, yerr=[err_low, err_high], fmt="none",
            ecolor="#333333", elinewidth=1.3, capsize=9, capthick=1.3, zorder=4)
for i, (bar, r, xa, na) in enumerate(zip(bars, rates, [x_A, x_B], [n_A, n_B])):
    ax.text(bar.get_x() + bar.get_width() / 2, r + err_high[i] + 0.45, f"{r:.2f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#222222")
    ax.text(bar.get_x() + bar.get_width() / 2, r / 2, f"{xa}/{na}",
            ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
ax.set_ylim(0, 11.8)
ax.set_ylabel(ylabel, fontsize=11)
ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="x", labelsize=10.5)
ax.tick_params(axis="y", labelsize=10)

info = ("测试信息：观察期 7 天 | 单侧双比例 z 检验 | α = 0.05 | z = 1.9584\n"
        "p(单侧) = 0.0251（拒绝原假设 H0）| Δ = +2.05 pp（+34.18%）| "
        "n_A = 1200, n_B = 1180 | 当前功效 ≈ 62.3%")
fig.text(0.5, 0.025, info, ha="center", va="bottom", fontsize=9.2, color="#333333",
         bbox=dict(boxstyle="round,pad=0.45", facecolor="#F5F7FA", edgecolor="#C9D2DE", linewidth=0.8))
plt.tight_layout(rect=[0, 0.135, 1, 1])
out_png = "C:/Users/zouju/WorkBuddy/2026-11-11-09-28-17/work/conversion_rate_chart.png"
plt.savefig(out_png, dpi=200, facecolor="white")
print(f"[8] 图表已保存      {out_png}")

