# 作业1：
$\boldsymbol{\hat\beta}_{Ridge} = \frac{1}{1+\lambda}\boldsymbol{\hat\beta}_{OLS}$

正交设计定义：设计矩阵$X$满足 $X^TX = I$（自变量之间正交，$X^TX$为单位矩阵）

Ridge回归目标函数：
$$
L(\beta) = \|y-X\beta\|_2^2+\lambda\|\beta\|_2^2
$$
对$\beta$求导并令导数等于0：
$$
\frac{\partial L}{\partial \beta}=-2X^T(y-X\beta)+2\lambda\beta = 0
$$
$$
X^TX\beta+\lambda\beta = X^Ty
$$
$$
(X^TX+\lambda I)\beta = X^Ty
$$
得到Ridge回归的解：
$$
\hat\beta_{Ridge}=(X^TX+\lambda I)^{-1}X^Ty
$$

OLS的解：
$$
\hat\beta_{OLS}=(X^TX)^{-1}X^Ty
$$

代入正交条件 $X^TX=I$：
$$
\hat\beta_{Ridge}=(I+\lambda I)^{-1}X^Ty=\big((1+\lambda)I\big)^{-1}X^Ty=\frac{1}{1+\lambda}I\cdot X^Ty
$$
而 $\hat\beta_{OLS}=I^{-1}X^Ty=X^Ty$，因此：
$$
\boldsymbol{\hat\beta}_{Ridge}=\frac{1}{1+\lambda}\boldsymbol{\hat\beta}_{OLS}
$$
证毕。

---

# 作业3：
Z-score标准化公式：
$$
x^*=\frac{x-\bar x}{\sigma}
$$

### 原因
Ridge、Lasso、ElasticNet的惩罚项 $\lambda\|\beta\|_2^2$ / $\lambda\|\beta\|_1$ 直接对回归系数$\beta$施加惩罚。
回归系数$\beta$的大小和自变量量纲、单位强相关。
举例：一个变量单位是元，取值几千；另一个变量是年龄，取值几十。同样变化1单位，含义完全不同，回归系数$\beta$天然量级差距巨大。

标准化之后：所有自变量均值为0，方差为1，全部自变量统一到相同尺度，惩罚对每一个变量公平，$\lambda$的约束效果不会被变量单位干扰。

### 不做标准化带来的后果
1. 量纲大的变量，对应的系数$\beta$天然偏小，惩罚几乎不起作用；
2. 量纲小的变量，对应的系数$\beta$天然偏大，惩罚会过度压缩该变量系数；
3. 交叉验证选出的最优$\lambda$是虚假的，模型倾向保留单位量级大的变量，变量选择结果失真；
4. Lasso会错误地将量纲小、但业务上很重要的变量压缩到0，错误剔除有效特征。

一句话总结：正则化惩罚对象是系数$\beta$，而系数受变量单位影响，所以正则化模型必须标准化消除量纲差异。普通OLS不需要标准化（单纯预测结果不变），但L1/L2正则模型必须标准化。