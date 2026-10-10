import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ========== 1. 加载Hitters数据集，预处理 ==========
Hitters = load_data('Hitters')
Hitters = Hitters.dropna()
Hitters = pd.get_dummies(Hitters, drop_first=True)
X = Hitters.drop('Salary', axis=1)
y = Hitters['Salary']
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
lambdas = np.logspace(3, -2, 100)

# ========== 2. 建立Pipeline：标准化 + 模型 ==========
ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=kfold))
])
ridge_pipe.fit(X, y)
lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", ElasticNetCV(alphas=lambdas, l1_ratio=1, cv=kfold))
])
lasso_pipe.fit(X, y)
elastic_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("elastic", ElasticNetCV(alphas=lambdas, l1_ratio=0.5, cv=kfold))
])
elastic_pipe.fit(X, y)

# ========== 3. 提取最优参数、系数、CV误差 ==========
ridge = ridge_pipe.named_steps['ridge']
lasso = lasso_pipe.named_steps['lasso']
elastic = elastic_pipe.named_steps['elastic']

print("="*60)
print(f"Ridge最优alpha(lambda): {ridge.alpha_:.4f}")
print(f"Lasso最优alpha(lambda): {lasso.alpha_:.4f}")
print(f"ElasticNet最优alpha(lambda): {elastic.alpha_:.4f}")
print("="*60)
print("Ridge系数（无稀疏，全部不为0）")
print(ridge.coef_)
print("Lasso系数（存在稀疏，部分系数压缩至0）")
print(lasso.coef_)
print("ElasticNet系数（部分稀疏）")
print(elastic.coef_)

# ========== 4. 绘制CV MSE路径图【这里已经修复维度bug】 ==========
neg_log_lambda = -np.log10(lambdas)
fig, ax = plt.subplots(1,3, figsize=(18,6))
ax[0].plot(neg_log_lambda, ridge.mse_path_)
ax[0].axvline(-np.log10(ridge.alpha_), color="k", linestyle="--")
ax[0].set_xlabel("-log10(lambda)")
ax[0].set_ylabel("CV MSE")
ax[0].set_title("Ridge CV MSE路径")
ax[1].plot(neg_log_lambda, lasso.mse_path_)
ax[1].axvline(-np.log10(lasso.alpha_), color="k", linestyle="--")
ax[1].set_xlabel("-log10(lambda)")
ax[1].set_title("Lasso CV MSE路径")
ax[2].plot(neg_log_lambda, elastic.mse_path_)
ax[2].axvline(-np.log10(elastic.alpha_), color="k", linestyle="--")
ax[2].set_xlabel("-log10(lambda)")
ax[2].set_title("ElasticNet CV MSE路径")

plt.tight_layout()
plt.show()

# ========== 5. 对比三种模型交叉验证MSE ==========
ridge_cv_mse = np.min(ridge.mse_path_.mean(axis=1))
lasso_cv_mse = np.min(lasso.mse_path_.mean(axis=1))
elastic_cv_mse = np.min(elastic.mse_path_.mean(axis=1))
print("="*60)
print(f"Ridge最小CV MSE: {ridge_cv_mse:.2f}")
print(f"Lasso最小CV MSE: {lasso_cv_mse:.2f}")
print(f"ElasticNet最小CV MSE: {elastic_cv_mse:.2f}")