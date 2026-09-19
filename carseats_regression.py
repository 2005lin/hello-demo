# -*- coding: utf-8 -*-
import sys
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
sys.stdout.reconfigure(encoding="utf-8")
# ---------- 1. 加载 Carseats 数据集 ----------
from ISLP import load_data
Carseats = load_data("Carseats")
print("数据形状:", Carseats.shape)
print("ShelveLoc 取值:", list(Carseats["ShelveLoc"].cat.categories))
# ---------- 2. 构建多元线性回归模型 ----------
formula = "Sales ~ Price + Income + Advertising + C(ShelveLoc)"
model = sm.OLS.from_formula(formula, data=Carseats).fit()
# ---------- 3. 拟合报告 ----------
print("\n" + "=" * 72)
print("多元线性回归拟合报告（OLS Summary）")
print("=" * 72)
print(model.summary())
# ---------- 4. 判断 ShelveLoc 的基准组 ----------
present = [c for c in model.params.index if "ShelveLoc" in c]          # 模型中出现的哑变量列
all_lv = list(Carseats["ShelveLoc"].cat.categories)                     # 全部水平
baseline = [lv for lv in all_lv if not any(lv in p for p in present)]  # 未出现的即基准组
print("\n【问题1】ShelveLoc 的哑变量:", present)
print("【问题1】ShelveLoc 的基准组 =", baseline[0],
      "（treatments 编码下未进入模型的水平即为参照基准）")
# ---------- 5. 解读 ShelveLoc[Good] 系数 ----------
good_key = [c for c in model.params.index if "Good" in c][0]
b_good, p_good = model.params[good_key], model.pvalues[good_key]
print("\n【问题2】ShelveLoc[Good] 系数解读")
print(f"   {good_key} = {b_good:.4f}，p 值 = {p_good:.3e}（高度显著）")
print("   商业含义：在价格(Price)、收入(Income)、广告(Advertising)相同的情况下，")
print(f"   货架位置为 Good 的门店，其年销售额比基准组 Bad 平均高出 {b_good:.2f} 千件，")
print("   说明『好货架位置』对儿童座椅销量有非常显著的正向提升作用。")
# ---------- 6. 计算 VIF，诊断多重共线性 ----------
print("\n【问题3】各自变量方差膨胀因子 VIF")
X, names = model.model.exog, model.model.exog_names
vif = pd.DataFrame({
    "变量": names,
    "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])],
})
print(vif.to_string(index=False))
print("   判断标准：VIF < 5 可接受；5~10 中度共线性；>10 严重多重共线性。")
num_vif = vif[vif["变量"] != "Intercept"]["VIF"]
if num_vif.max() < 5:
    print(f"   结论：各自变量 VIF 最大仅为 {num_vif.max():.2f}，远小于 5，")
    print("          不存在明显的多重共线性风险，模型系数估计稳定可靠。")
