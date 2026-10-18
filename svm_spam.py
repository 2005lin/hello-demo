# -*- coding:utf-8 -*-
"""
SVM 垃圾邮件分类 Spambase数据集
参考SVM课件案例①
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 1.加载数据 ======================
# 如果kaggle下载csv，替换文件名spambase.csv
# 也可以直接读取UCI在线数据
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
columns = [f"feat_{i}" for i in range(57)] + ["label"]
df = pd.read_csv(url, names=columns)

print("数据集形状：", df.shape)
print("标签分布：")
print(df["label"].value_counts())

X = df.drop("label", axis=1)
y = df["label"]

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ====================== 2.特征标准化 SVM必须标准化！ ======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================== 3.训练线性SVM ======================
print("\n========== 线性核SVM C=1 ==========")
linear_svm = LinearSVC(C=1, max_iter=10000, random_state=42)
linear_svm.fit(X_train_scaled, y_train)

y_pred_linear = linear_svm.predict(X_test_scaled)
acc_linear = accuracy_score(y_test, y_pred_linear)
print(f"测试集准确率：{acc_linear:.4f}")
print(classification_report(y_test, y_pred_linear))

# ====================== 4.RBF高斯核SVM +网格搜索调参 ======================
# 课件推荐搜索范围 C ∈ [1e-3,1e3], gamma ∈ [1e-4,1e1]
param_grid = {
    "C": [0.001, 0.1, 1, 10, 100],
    "gamma": [0.0001, 0.001, 0.01, 0.1, 1]
}

rbf_svm = SVC(kernel="rbf", random_state=42)
grid = GridSearchCV(rbf_svm, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train_scaled, y_train)

print("\n========== RBF核SVM网格搜索最佳参数 ==========")
print("最佳参数：", grid.best_params_)
print("交叉验证最佳得分：", grid.best_score_)

best_rbf = grid.best_estimator_
y_pred_rbf = best_rbf.predict(X_test_scaled)
acc_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"RBF测试集准确率：{acc_rbf:.4f}")
print(classification_report(y_test, y_pred_rbf))

# ====================== 5.混淆矩阵可视化 ======================
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    classes = ["正常邮件","垃圾邮件"]
    plt.xticks([0,1], classes)
    plt.yticks([0,1], classes)
    for i in range(2):
        for j in range(2):
            plt.text(j,i,str(cm[i][j]),ha="center",va="center",fontsize=14)
    plt.ylabel("真实标签")
    plt.xlabel("预测标签")
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, y_pred_linear, "线性SVM混淆矩阵")
plot_confusion_matrix(y_test, y_pred_rbf, "RBF核SVM混淆矩阵")

# ====================== 6.结果汇总对比 ======================
result_df = pd.DataFrame({
    "模型":["线性SVM(C=1)","RBF最优SVM"],
    "测试集准确率":[acc_linear, acc_rbf]
})
print("\n=====模型性能汇总=====")
print(result_df)
