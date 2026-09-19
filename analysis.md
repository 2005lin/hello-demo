## 一
一元线性回归拟合方程：$\hat y_i=\hat\beta_0+\hat\beta_1 x_i$，残差定义：$e_i=y_i-\hat y_i$。
最小二乘法目标是最小化残差平方和 $RSS=\sum_{i=1}^n e_i^2=\sum_{i=1}^n(y_i-\hat\beta_0-\hat\beta_1x_i)^2$。
对截距$\hat\beta_0$、斜率$\hat\beta_1$分别求偏导并令偏导数等于0，得到正规方程组：
$$
\begin{cases}
\displaystyle \frac{\partial RSS}{\partial \hat\beta_0} = -2\sum_{i=1}^n (y_i-\hat\beta_0-\hat\beta_1 x_i)=0 \\[6pt]
\displaystyle \frac{\partial RSS}{\partial \hat\beta_1} = -2\sum_{i=1}^n x_i(y_i-\hat\beta_0-\hat\beta_1 x_i)=0
\end{cases}
$$

（1）证明$\boldsymbol{\sum_{i=1}^n e_i=0}$
将$e_i=y_i-\hat\beta_0-\hat\beta_1 x_i$代入第一个方程：
$$
\sum_{i=1}^n (y_i-\hat\beta_0-\hat\beta_1 x_i)=\sum_{i=1}^n e_i=0
$$
可得残差之和等于0。

（2）证明$\boldsymbol{\sum_{i=1}^n x_i e_i=0}$
代入第二个方程：
$$
\sum_{i=1}^n x_i(y_i-\hat\beta_0-\hat\beta_1 x_i)=\sum_{i=1}^n x_i e_i=0
$$
可得残差与自变量乘积之和等于0。
证毕。

## 二
相关公式：
$R^2=1-\dfrac{RSS}{TSS},\quad R^2_{adj}=1-\dfrac{RSS/(n-k-1)}{TSS/(n-1)}$
$RSS$为残差平方和，$TSS$为总平方和，$n$为样本量，$k$为自变量个数，$n-k-1$为模型自由度。

1）$R^2$上升或保持不变的原因：
$TSS$仅由响应变量$y$决定，新增自变量不会改变$TSS$。增加任意自变量（即使是纯随机无关特征），OLS会利用该变量拟合样本随机噪声，**残差平方和RSS只会减小或不变**。由$R^2$公式，RSS减小则$R^2$上升或不变，$R^2$不会对增加变量进行惩罚。

2）$R^2_{adj}$下降的原因：
$R^2_{adj}$引入**自由度惩罚**。每新增一个自变量，自变量个数$k$增加，模型自由度$n-k-1$减小。对于纯随机无关特征，RSS下降幅度很小，自由度减小带来的影响占主导，$\dfrac{RSS}{n-k-1}$变大，最终使$R^2_{adj}$下降。

总结：普通$R^2$不惩罚多余变量；调整$R^2_{adj}$通过自由度惩罚，避免盲目增加无效自变量。
