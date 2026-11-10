

# -*- coding: utf-8 -*-
import json
import math
import os
 
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
 
BASE = os.path.dirname(os.path.abspath(__file__))
np.random.seed(6000)
 

n_A, x_A = 1200, 72
n_B, x_B = 1180, 95
ALPHA = 0.05            # one-sided
ALPHA_TWO = 0.05        # two-sided reference
 
p_A = x_A / n_A
p_B = x_B / n_B
diff = p_B - p_A
rel_lift = diff / p_A
 
R = {}
R["data"] = dict(n_A=n_A, x_A=x_A, n_B=n_B, x_B=x_B, alpha=ALPHA)
R["rates"] = dict(p_A=p_A, p_B=p_B, diff=diff, rel_lift=rel_lift,
                  lift_pp=diff * 100)
 

p_pool = (x_A + x_B) / (n_A + n_B)
se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_A + 1 / n_B))
z_stat = diff / se_pool
p_one = 1 - stats.norm.cdf(z_stat)          # one-sided (right tail)
p_two = 2 * (1 - stats.norm.cdf(abs(z_stat)))
z_crit_one = stats.norm.ppf(1 - ALPHA)
z_crit_two = stats.norm.ppf(1 - ALPHA_TWO / 2)
 
R["ztest"] = dict(p_pool=p_pool, se_pool=se_pool, z_stat=z_stat,
                  p_one_sided=p_one, p_two_sided=p_two,
                  z_crit_one_sided=z_crit_one, z_crit_two_sided=z_crit_two,
                  reject_one_sided=bool(p_one < ALPHA),
                  reject_two_sided=bool(p_two < ALPHA_TWO))
 

z_sm, p_sm = proportions_ztest(count=np.array([x_B, x_A]),
                               nobs=np.array([n_B, n_A]),
                               alternative="larger")
R["ztest"]["statsmodels_z"] = float(z_sm)
R["ztest"]["statsmodels_p_one_sided"] = float(p_sm)
 

table = np.array([[x_B, n_B - x_B], [x_A, n_A - x_A]])
chi2, p_chi, dof, expected = stats.chi2_contingency(table, correction=False)
chi2_y, p_chi_y, _, _ = stats.chi2_contingency(table, correction=True)
R["chi2"] = dict(table=table.tolist(), expected=expected.tolist(),
                 chi2=float(chi2), p_value=float(p_chi), dof=int(dof),
                 chi2_yates=float(chi2_y), p_value_yates=float(p_chi_y),
                 z_squared=float(z_stat ** 2),
                 identity_holds=bool(abs(chi2 - z_stat ** 2) < 1e-8))
 

odds, p_fisher = stats.fisher_exact(table, alternative="greater")
R["fisher"] = dict(odds_ratio=float(odds), p_one_sided=float(p_fisher))
 

se_unpool = math.sqrt(p_A * (1 - p_A) / n_A + p_B * (1 - p_B) / n_B)
ci95_low = diff - z_crit_two * se_unpool
ci95_high = diff + z_crit_two * se_unpool
lower_onesided_95 = diff - z_crit_one * se_unpool

se_log_rr = math.sqrt((1 - p_B) / (n_B * p_B) + (1 - p_A) / (n_A * p_A))
rr = p_B / p_A
rr_low = math.exp(math.log(rr) - z_crit_two * se_log_rr)
rr_high = math.exp(math.log(rr) + z_crit_two * se_log_rr)
 
R["ci"] = dict(se_unpooled=se_unpool,
               ci95_diff=[ci95_low, ci95_high],
               one_sided_lower_95=lower_onesided_95,
               relative_risk=rr,
               ci95_rel_lift=[rr_low - 1, rr_high - 1])
 

R["ci"]["wilson_A"] = list(proportion_confint(x_A, n_A, alpha=0.05, method="wilson"))
R["ci"]["wilson_B"] = list(proportion_confint(x_B, n_B, alpha=0.05, method="wilson"))
 

effect_h = proportion_effectsize(p_B, p_A)          # Cohen's h
power_analysis = NormalIndPower()
achieved_power = power_analysis.power(effect_size=abs(effect_h),
                                     nobs1=n_A, ratio=n_B / n_A,
                                     alpha=ALPHA, alternative="larger")
R["power"] = dict(cohens_h=float(effect_h),
                  achieved_power_one_sided=float(achieved_power),
                  interpretation="P(reject H0 | true effect = observed effect)")
 
 
def required_n(p1, p2, alpha=0.05, target_power=0.80, ratio=1.0):
    """Closed-form normal approximation for two independent proportions."""
    pbar = (p1 + p2) / 2
    za = stats.norm.ppf(1 - alpha)          # one-sided
    zb = stats.norm.ppf(target_power)
    num = (za * math.sqrt(2 * pbar * (1 - pbar))
           + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p1) ** 2)
 
 
scenarios = []
for label, mde_pp in [("Observed effect (+2.05 pp)", diff * 100),
                      ("MDE = +1.0 pp", 1.0),
                      ("MDE = +1.5 pp", 1.5),
                      ("MDE = +2.0 pp", 2.0)]:
    p2 = p_A + mde_pp / 100
    n_req = required_n(p_A, p2)
    h = proportion_effectsize(p2, p_A)
    n_req_sm = math.ceil(power_analysis.solve_power(
        effect_size=abs(h), alpha=ALPHA, power=0.80, ratio=1.0,
        alternative="larger"))
    scenarios.append(dict(label=label, mde_pp=mde_pp, p2=p2,
                          n_per_group_closed_form=n_req,
                          n_per_group_statsmodels=n_req_sm,
                          total=2 * n_req))
R["power"]["sample_size_scenarios"] = scenarios
R["power"]["observed_n_per_group"] = n_A
R["power"]["shortfall_ratio_observed_effect"] = (
    scenarios[0]["n_per_group_closed_form"] / n_A)
 
 

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
from matplotlib import gridspec
 
fig = plt.figure(figsize=(12.4, 6.0), dpi=180)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.65, 1.0], wspace=0.06, figure=fig)
ax = fig.add_subplot(gs[0, 0])
ax_info = fig.add_subplot(gs[0, 1])
ax_info.axis("off")
 
groups = ["A: Control\n(original UI)", "B: Treatment\n(redesigned)"]
rates = [p_A * 100, p_B * 100]
errs = [z_crit_two * math.sqrt(p_A * (1 - p_A) / n_A) * 100,
        z_crit_two * math.sqrt(p_B * (1 - p_B) / n_B) * 100]
colors = ["#8C9BAB", "#C0392B"]
bars = ax.bar(groups, rates, width=0.5, color=colors,
              yerr=errs, capsize=8, edgecolor="#333333", linewidth=0.8,
              error_kw=dict(elinewidth=1.4, ecolor="#333333"), zorder=3)
 
for i, (b, r, n, x) in enumerate(zip(bars, rates, [n_A, n_B], [x_A, x_B])):
    ax.text(b.get_x() + b.get_width() / 2, r + errs[i] + 0.22,
            f"{r:.2f}%  ({x}/{n})", ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color="#222222")
 
ax.set_ylabel("Purchase conversion rate (%)", fontsize=12)
ax.set_title("A/B Test: One-Week Purchase Conversion Rate\n"
             "Error bars = 95% normal-approximation CI (Wald)",
             fontsize=13, fontweight="bold")
YMAX = 13.2
ax.set_ylim(0, YMAX)
ax.set_yticks(range(0, 14, 2))
ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax.set_axisbelow(True)
 
# lift bracket
y_br = 11.2
ax.annotate("", xy=(1, y_br), xytext=(0, y_br),
            arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.7))
ax.text(0.5, y_br + 0.22, f"+{diff*100:.2f} pp absolute lift",
        ha="center", color="#C0392B", fontweight="bold", fontsize=11.5)
 
# separate information panel
ann = (f"TEST INFORMATION\n"
       f"------------------------------\n"
       f"n_A = {n_A},   n_B = {n_B}\n"
       f"x_A = {x_A},     x_B = {x_B}\n"
       f"------------------------------\n"
       f"Lift  +{diff*100:.2f} pp\n"
       f"      +{rel_lift*100:.1f}% relative\n"
       f"------------------------------\n"
       f"Pooled z      = {z_stat:.4f}\n"
       f"One-sided p   = {p_one:.4f}\n"
       f"Two-sided p   = {p_two:.4f}\n"
       f"chi2 (1 df)   = {chi2:.4f}\n"
       f"------------------------------\n"
       f"alpha = 0.05 (one-sided)\n"
       f"Decision: REJECT H0\n"
       f"* two-sided 0.0502 > 0.05\n"
       f"* 95% CI [{ci95_low*100:+.2f}, {ci95_high*100:+.2f}] pp\n"
       f"* power = {achieved_power:.1%} (< 80%)")
ax_info.text(0.02, 0.97, ann, transform=ax_info.transAxes,
             ha="left", va="top", fontsize=10.2, family="monospace",
             linespacing=1.62,
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#F5F7FA",
                       edgecolor="#8C9BAB", alpha=0.97))
 
fig.tight_layout()
f1 = os.path.join(BASE, "figure1_conversion.png")
fig.savefig(f1, bbox_inches="tight", facecolor="white")
plt.close(fig)
 
# ----------------------------------------------------------------------------
# 6. Figure 2 - power curves vs sample size
# ----------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9.2, 5.2), dpi=180)
ns = np.linspace(200, 14000, 400)
for mde, col, lab in [(diff * 100, "#C0392B", f"Observed +{diff*100:.2f} pp"),
                      (1.0, "#2C6FA6", "MDE +1.0 pp"),
                      (1.5, "#2E8B57", "MDE +1.5 pp"),
                      (2.0, "#8E5BB7", "MDE +2.0 pp")]:
    h = abs(proportion_effectsize(p_A + mde / 100, p_A))
    pw = power_analysis.power(effect_size=h, nobs1=ns, ratio=1.0,
                              alpha=ALPHA, alternative="larger")
    ax2.plot(ns, pw, lw=2.1, color=col, label=lab)
 
ax2.axhline(0.80, ls="--", color="#555555", lw=1.2, label="Target power = 0.80")
ax2.axvline(n_A, ls=":", color="#888888", lw=1.5,
            label=f"Actual n/group ~ {n_A}")
for s in R["power"]["sample_size_scenarios"][:1] + R["power"]["sample_size_scenarios"][2:3]:
    ax2.axvline(s["n_per_group_closed_form"], ls=":", color="#C0392B", alpha=0.45, lw=1.0)
 
ax2.set_xlabel("Sample size per group (n)", fontsize=12)
ax2.set_ylabel("Statistical power (1 - beta)", fontsize=12)
ax2.set_title("Power Analysis: one-sided alpha = 0.05, baseline p_A = 6.00%",
              fontsize=13, fontweight="bold")
ax2.set_ylim(0, 1.03)
ax2.yaxis.grid(True, ls="--", alpha=0.4)
ax2.set_axisbelow(True)
ax2.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
fig2.tight_layout()
f2 = os.path.join(BASE, "figure2_power.png")
fig2.savefig(f2, bbox_inches="tight", facecolor="white")
plt.close(fig2)
 
R["figures"] = [f1, f2]
 

with open(os.path.join(BASE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, indent=2, ensure_ascii=False)
 
lines = []
lines.append("=== DSCI 6000 GROUP ASSIGNMENT 1 - NUMERICAL OUTPUT ===")
lines.append(f"Group A: {x_A}/{n_A} = {p_A:.6f} ({p_A*100:.4f}%)")
lines.append(f"Group B: {x_B}/{n_B} = {p_B:.6f} ({p_B*100:.4f}%)")
lines.append(f"Absolute lift = {diff:.6f} ({diff*100:.4f} pp)")
lines.append(f"Relative lift = {rel_lift*100:.4f}%")
lines.append("")
lines.append(f"Pooled p = {p_pool:.8f};  SE_pooled = {se_pool:.8f}")
lines.append(f"z = {z_stat:.6f};  z_crit(one,0.05) = {z_crit_one:.6f}")
lines.append(f"p one-sided = {p_one:.6f}  -> reject H0: {p_one < ALPHA}")
lines.append(f"p two-sided = {p_two:.6f}  -> reject H0: {p_two < ALPHA_TWO}")
lines.append(f"statsmodels z = {z_sm:.6f}, p = {p_sm:.6f}")
lines.append("")
lines.append(f"Chi2 (no correction) = {chi2:.6f}, p = {p_chi:.6f}, dof = {dof}")
lines.append(f"Chi2 (Yates)         = {chi2_y:.6f}, p = {p_chi_y:.6f}")
lines.append(f"z^2 = {z_stat**2:.6f}  (equals chi2: {abs(chi2-z_stat**2)<1e-8})")
lines.append(f"Fisher exact one-sided p = {p_fisher:.6f}; OR = {odds:.6f}")
lines.append("")
lines.append(f"SE_unpooled = {se_unpool:.8f}")
lines.append(f"95% CI diff (pp) = [{ci95_low*100:+.4f}, {ci95_high*100:+.4f}]")
lines.append(f"One-sided 95% lower bound (pp) = {lower_onesided_95*100:+.4f}")
lines.append(f"Relative risk = {rr:.6f}; 95% CI rel lift = [{(rr_low-1)*100:+.3f}%, {(rr_high-1)*100:+.3f}%]")
lines.append(f"Wilson A = {R['ci']['wilson_A']}")
lines.append(f"Wilson B = {R['ci']['wilson_B']}")
lines.append("")
lines.append(f"Cohen's h = {effect_h:.6f}")
lines.append(f"Achieved power (observed effect, one-sided) = {achieved_power:.4f}")
lines.append("")
lines.append("Minimum sample size per group (alpha=0.05 one-sided, power=0.80):")
for s in scenarios:
    lines.append(f"  {s['label']:<32} p2={s['p2']:.5f}  n/group={s['n_per_group_closed_form']:>7,}  "
                 f"(statsmodels {s['n_per_group_statsmodels']:>7,})  total={s['total']:,}")
lines.append("")
lines.append("Figures (written next to this script): " +
             ", ".join(os.path.basename(p) for p in R["figures"]))
txt = "\n".join(lines)
with open(os.path.join(BASE, "results.txt"), "w", encoding="utf-8") as f:
    f.write(txt)
print(txt)
