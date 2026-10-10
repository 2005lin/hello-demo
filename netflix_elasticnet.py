import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 读取Netflix数据集 =====================
df = pd.read_csv("netflix_titles.csv")
print("原始数据形状：", df.shape)

# ===================== 2. 数据清洗（只保留电影，预测电影时长duration） =====================
# 只选Movie，剔除TV Show
df = df[df['type'] == 'Movie'].copy()

# 处理duration：原始是 "90 min"，提取数字
def extract_minutes(duration_str):
    if pd.isna(duration_str):
        return np.nan
    return int(duration_str.replace(" min", ""))
df['duration_min'] = df['duration'].apply(extract_minutes)

# 删除缺失值
df = df.dropna(subset=['duration_min', 'country', 'release_year', 'rating'])

# 选取我们要用的特征
feature_cols = ["release_year", "country", "rating"]
X_raw = df[feature_cols]
y = df["duration_min"] # 目标变量：预测电影分钟时长

# 分类特征做独热编码
X = pd.get_dummies(X_raw, drop_first=True)
print("编码后特征数量：", X.shape[1])
print("用于建模样本数量：", len(X))

# ===================== 3. 划分训练集、测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== 4. 建立管道 + 网格搜索寻找ElasticNet最优超参数【加分部分】 =====================
# 5折交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 网格搜索范围：alpha正则强度，l1_ratio控制L1/L2比重
elastic_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNet())
])
param_grid = {
    "model__alpha": [0.01, 0.1, 1, 10, 100],
    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}
grid_search = GridSearchCV(elastic_pipe, param_grid, cv=kfold, scoring="neg_mean_squared_error")
grid_search.fit(X_train, y_train)

print("\n======== ElasticNet网格搜索最优超参数 ========")
print(f"最优alpha: {grid_search.best_params_['model__alpha']}")
print(f"最优l1_ratio: {grid_search.best_params_['model__l1_ratio']}")
print(f"最优交叉验证MSE: {-grid_search.best_score_:.2f}")

# 三个模型
models = {
    "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "Lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=1.0))]),
    "ElasticNet_Best": grid_search.best_estimator_ # 使用网格搜索找到的最优ElasticNet
}

# ===================== 5. 交叉验证评估 =====================
print("\n===== 5折交叉验证结果（MSE）=====")
for name, pipe in models.items():
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=kfold, scoring="neg_mean_squared_error")
    cv_mse = -np.mean(cv_scores)
    pipe.fit(X_train, y_train)
    test_pred = pipe.predict(X_test)
    test_mse = np.mean((y_test - test_pred)**2)
    print(f"{name:15s} | 交叉验证MSE: {cv_mse:8.2f} | 测试集MSE: {test_mse:8.2f}")

# ===================== 6. ElasticNet 系数可视化【优化标签，不会文字挤在一起】 =====================
elastic_model = models["ElasticNet_Best"]
elastic_model.fit(X_train, y_train)
coefs = elastic_model.named_steps["model"].coef_
feature_names = X.columns

# 筛选非0系数画图
coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
coef_df = coef_df[coef_df["coef"] != 0].sort_values("coef")

# 精简过长特征名称
def short_name(s):
    if "country_" in s:
        return s.replace("country_","")[:22]
    return s
coef_df["feature"] = coef_df["feature"].apply(short_name)

plt.figure(figsize=(11,8))
plt.barh(coef_df["feature"], coef_df["coef"])
plt.title("ElasticNet 各特征回归系数（最优参数）", fontsize=14)
plt.xlabel("系数大小", fontsize=12)
plt.tight_layout()
plt.savefig("elasticnet_coef.png", dpi=300) # 图片保存到本地文件夹
plt.show()

print("\n==== ElasticNet非零特征系数 ====")
print(coef_df)