# 机器学习学习笔记（全面版）

## 一、什么是机器学习？

机器学习就是让计算机从数据中**自动学习规律**，而不是人为编写规则。

**举个例子**：
- 传统编程：告诉电脑"如果邮件包含'中奖'就是垃圾邮件"
- 机器学习：给电脑1万封已标记的邮件，让它自己总结规律

---

## 二、机器学习的三大类型

### 1. 监督学习（Supervised Learning）
有"标准答案"的学习方式，像做有答案的练习题。

**典型任务**：
- **分类**：判断邮件是否垃圾邮件（离散值）
- **回归**：预测房价（连续值）

### 2. 无监督学习（Unsupervised Learning）
没有标准答案，让机器自己发现数据的结构。

**典型任务**：
- **聚类**：把顾客分成不同群体
- **降维**：把高维数据压缩成低维

### 3. 强化学习（Reinforcement Learning）
通过奖励和惩罚学习，像训练宠物。

**典型应用**：AlphaGo、自动驾驶

---

## 三、核心概念

### 3.1 特征（Features）与标签（Labels）

```
数据样本 = 特征 + 标签

例：房价预测
特征：面积、房间数、地段 → X
标签：价格 → y
```

### 3.2 损失函数（Loss Function）

衡量模型预测值与真实值的差距。

**均方误差（MSE）**：回归问题常用
$$L_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**交叉熵损失（Cross-Entropy）**：分类问题常用
$$L_{CE} = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

### 3.3 梯度下降（Gradient Descent）

优化算法，用于找到损失函数的最小值。

**更新规则**：
$$\theta = \theta - \alpha \frac{\partial L}{\partial \theta}$$

**三种变体**：
- **批量梯度下降（BGD）**：使用全部数据
- **随机梯度下降（SGD）**：每次使用一个样本
- **小批量梯度下降（Mini-batch GD）**：每次使用一小批样本

### 3.4 正则化（Regularization）

防止过拟合的技术。

**L1正则化（Lasso）**：
$$L = L_{original} + \lambda\sum_{i=1}^{n}|\theta_i|$$

**L2正则化（Ridge）**：
$$L = L_{original} + \lambda\sum_{i=1}^{n}\theta_i^2$$

---

## 四、线性回归（Linear Regression）

### 4.1 原理

假设变量之间存在线性关系：
$$y = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n + \epsilon$$

**正规方程解**：
$$\theta = (X^TX)^{-1}X^Ty$$

### 4.2 Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. 生成模拟数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 普通线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 4. Ridge回归（L2正则化）
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# 5. Lasso回归（L1正则化）
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# 6. 评估
print("线性回归:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_lr):.4f}")
print(f"  R²: {r2_score(y_test, y_pred_lr):.4f}")
print(f"  参数: y = {lr.intercept_[0]:.2f} + {lr.coef_[0][0]:.2f}x")

print("\nRidge回归:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_ridge):.4f}")
print(f"  R²: {r2_score(y_test, y_pred_ridge):.4f}")

print("\nLasso回归:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_lasso):.4f}")
print(f"  R²: {r2_score(y_test, y_pred_lasso):.4f}")

# 7. 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X_test, y_test, color='blue', alpha=0.5)
plt.plot(X_test, y_pred_lr, color='red', linewidth=2)
plt.title('线性回归')
plt.xlabel('X')
plt.ylabel('y')

plt.subplot(1, 3, 2)
plt.scatter(X_test, y_test, color='blue', alpha=0.5)
plt.plot(X_test, y_pred_ridge, color='green', linewidth=2)
plt.title('Ridge回归')
plt.xlabel('X')

plt.subplot(1, 3, 3)
plt.scatter(X_test, y_test, color='blue', alpha=0.5)
plt.plot(X_test, y_pred_lasso, color='purple', linewidth=2)
plt.title('Lasso回归')
plt.xlabel('X')

plt.tight_layout()
plt.show()
```

---

## 五、逻辑回归（Logistic Regression）

### 5.1 原理

用于**二分类问题**，核心是**Sigmoid函数**：
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

预测概率：
$$P(y=1|x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

**损失函数**（对数损失）：
$$L = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

### 5.2 Python实现

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

# 1. 生成数据
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 训练模型
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 3. 预测
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# 4. 评估
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 5. ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
plt.xlabel('假正例率 (FPR)')
plt.ylabel('真正例率 (TPR)')
plt.title('ROC曲线')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 六、支持向量机（Support Vector Machine, SVM）

### 6.1 原理

寻找一个**最优超平面**，使得两类样本的**间隔最大化**。

**决策函数**：
$$f(x) = \text{sign}(w^Tx + b)$$

**优化目标**（硬间隔）：
$$\min_{w,b} \frac{1}{2}||w||^2$$
$$\text{s.t. } y_i(w^Tx_i + b) \geq 1, \forall i$$

**软间隔SVM**（允许一些错误）：
$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$
$$\text{s.t. } y_i(w^Tx_i + b) \geq 1-\xi_i, \xi_i \geq 0$$

其中$C$是惩罚参数，$\xi_i$是松弛变量。

**核技巧（Kernel Trick）**：
将数据映射到高维空间，使其线性可分。

常用核函数：
- **线性核**：$K(x_i, x_j) = x_i^Tx_j$
- **多项式核**：$K(x_i, x_j) = (x_i^Tx_j + c)^d$
- **RBF核（高斯核）**：$K(x_i, x_j) = \exp(-\gamma||x_i - x_j||^2)$
- **Sigmoid核**：$K(x_i, x_j) = \tanh(\alpha x_i^Tx_j + c)$

### 6.2 Python实现

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles

# 1. 生成非线性可分数据
X, y = make_circles(n_samples=500, noise=0.1, factor=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 线性SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
print(f"线性SVM准确率: {svm_linear.score(X_test_scaled, y_test):.4f}")

# 4. RBF核SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)
print(f"RBF核SVM准确率: {svm_rbf.score(X_test_scaled, y_test):.4f}")

# 5. 多项式核SVM
svm_poly = SVC(kernel='poly', degree=3, C=1.0)
svm_poly.fit(X_train_scaled, y_train)
print(f"多项式核SVM准确率: {svm_poly.score(X_test_scaled, y_test):.4f}")

# 6. 可视化决策边界
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title(title)

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plot_decision_boundary(svm_linear, X_train_scaled, y_train, '线性SVM')
plt.subplot(1, 3, 2)
plot_decision_boundary(svm_rbf, X_train_scaled, y_train, 'RBF核SVM')
plt.subplot(1, 3, 3)
plot_decision_boundary(svm_poly, X_train_scaled, y_train, '多项式核SVM')
plt.tight_layout()
plt.show()

# 7. 参数调优
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
print(f"测试集分数: {grid_search.score(X_test_scaled, y_test):.4f}")
```

---

## 七、决策树（Decision Tree）

### 7.1 原理

通过一系列if-else问题来分类或回归。

**信息熵（Entropy）**：
$$H(S) = -\sum_{i=1}^{c}p_i\log_2(p_i)$$

**基尼不纯度（Gini Impurity）**：
$$Gini(S) = 1 - \sum_{i=1}^{c}p_i^2$$

**信息增益（Information Gain）**：
$$IG(S, A) = H(S) - \sum_{v\in Values(A)}\frac{|S_v|}{|S|}H(S_v)$$

### 7.2 Python实现

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.datasets import load_iris

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 训练决策树（不同标准）
tree_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

tree_entropy.fit(X_train, y_train)
tree_gini.fit(X_train, y_train)

print(f"信息熵决策树准确率: {tree_entropy.score(X_test, y_test):.4f}")
print(f"基尼系数决策树准确率: {tree_gini.score(X_test, y_test):.4f}")

# 3. 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(tree_entropy, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('决策树（信息熵）')
plt.show()

# 4. 特征重要性
importances = tree_entropy.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n特征重要性:")
for i in range(len(importances)):
    print(f"{iris.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# 5. 决策树回归示例
X_reg = np.sort(5 * np.random.rand(80, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.randn(80) * 0.1

tree_reg = DecisionTreeRegressor(max_depth=5)
tree_reg.fit(X_reg, y_reg)

X_test_reg = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred_reg = tree_reg.predict(X_test_reg)

plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, s=20, edgecolor="black", c="darkorange", label="数据")
plt.plot(X_test_reg, y_pred_reg, color="cornflowerblue", linewidth=2, label="预测")
plt.xlabel("X")
plt.ylabel("y")
plt.title("决策树回归")
plt.legend()
plt.show()
```

---

## 八、K近邻算法（K-Nearest Neighbors）

### 8.1 原理

根据最近的K个邻居的类别投票决定。

**距离度量**：
- **欧氏距离**：$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$
- **曼哈顿距离**：$d(x, y) = \sum_{i=1}^{n}|x_i - y_i|$
- **闵可夫斯基距离**：$d(x, y) = (\sum_{i=1}^{n}|x_i - y_i|^p)^{1/p}$

### 8.2 Python实现

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# 1. 分类任务
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 2. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 不同距离度量
knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_minkowski = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)

knn_euclidean.fit(X_train_scaled, y_train)
knn_manhattan.fit(X_train_scaled, y_train)
knn_minkowski.fit(X_train_scaled, y_train)

print(f"欧氏距离KNN: {knn_euclidean.score(X_test_scaled, y_test):.4f}")
print(f"曼哈顿距离KNN: {knn_manhattan.score(X_test_scaled, y_test):.4f}")
print(f"闵可夫斯基距离KNN: {knn_minkowski.score(X_test_scaled, y_test):.4f}")

# 4. 寻找最佳K值
k_range = range(1, 31)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

best_k = k_range[np.argmax(scores)]
print(f"\n最佳K值: {best_k}, 准确率: {max(scores):.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_range, scores, marker='o')
plt.xlabel('K值')
plt.ylabel('准确率')
plt.title('不同K值的模型性能')
plt.axvline(best_k, color='r', linestyle='--', label=f'最佳K={best_k}')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 九、朴素贝叶斯（Naive Bayes）

### 9.1 原理

基于贝叶斯定理和特征独立性假设。

**贝叶斯定理**：
$$P(y|x) = \frac{P(x|y)P(y)}{P(x)}$$

**朴素贝叶斯分类器**：
$$\hat{y} = \arg\max_y P(y)\prod_{i=1}^{n}P(x_i|y)$$

**三种常见类型**：
- **高斯朴素贝叶斯**：特征服从正态分布
- **多项式朴素贝叶斯**：适用于离散计数（文本分类）
- **伯努利朴素贝叶斯**：特征是二值的

### 9.2 Python实现

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 1. 高斯朴素贝叶斯（连续特征）
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"高斯朴素贝叶斯准确率: {gnb.score(X_test, y_test):.4f}")

# 2. 文本分类示例（多项式朴素贝叶斯）
# 加载新闻数据
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

# TF-IDF特征提取
vectorizer = TfidfVectorizer(max_features=1000)
X_train_text = vectorizer.fit_transform(newsgroups_train.data)
X_test_text = vectorizer.transform(newsgroups_test.data)

# 多项式朴素贝叶斯
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_text, newsgroups_train.target)
print(f"多项式朴素贝叶斯（文本分类）准确率: {mnb.score(X_test_text, newsgroups_test.target):.4f}")

# 3. 伯努利朴素贝叶斯（二值特征）
X_binary = (X_train > X_train.mean()).astype(int)
X_test_binary = (X_test > X_test.mean()).astype(int)

bnb = BernoulliNB()
bnb.fit(X_binary, y_train)
print(f"伯努利朴素贝叶斯准确率: {bnb.score(X_test_binary, y_test):.4f}")
```

---

## 十、集成学习（Ensemble Learning）

### 10.1 Bagging（Bootstrap Aggregating）

通过自助采样训练多个模型，投票决定最终结果。

**随机森林（Random Forest）**：
- Bagging + 随机特征选择
- 训练多棵决策树，每棵树只看部分特征

```python
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

# 1. 随机森林
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
print(f"随机森林准确率: {rf.score(X_test, y_test):.4f}")

# 2. 特征重要性
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [iris.feature_names[i] for i in indices], rotation=45)
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.show()

# 3. Bagging决策树
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    random_state=42
)
bagging.fit(X_train, y_train)
print(f"Bagging准确率: {bagging.score(X_test, y_test):.4f}")
```

### 10.2 Boosting

串行训练多个弱学习器，每个学习器关注前一个的错误。

**AdaBoost**：
权重更新公式：
$$w_i^{(t+1)} = w_i^{(t)} \exp(-\alpha_t y_i h_t(x_i))$$

**Gradient Boosting**：
每次拟合前一次的残差。

**XGBoost**（eXtreme Gradient Boosting）：
优化的梯度提升算法，加入正则化。

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
print(f"AdaBoost准确率: {ada.score(X_test, y_test):.4f}")

# 2. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                max_depth=3, random_state=42)
gb.fit(X_train, y_train)
print(f"Gradient Boosting准确率: {gb.score(X_test, y_test):.4f}")

# 3. XGBoost
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, 
                    max_depth=3, random_state=42)
xgb.fit(X_train, y_train)
print(f"XGBoost准确率: {xgb.score(X_test, y_test):.4f}")

# 4. 性能对比
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = model.score(X_test, y_test)

plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values())
plt.ylabel('准确率')
plt.title('不同模型性能对比')
plt.xticks(rotation=45)
plt.ylim(0.8, 1.0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```

### 10.3 Stacking（堆叠）

用多个基模型的预测作为新特征，训练元模型。

```python
from sklearn.ensemble import StackingClassifier

# 定义基模型
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# 定义元模型
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

stacking.fit(X_train_scaled, y_train)
print(f"Stacking准确率: {stacking.score(X_test_scaled, y_test):.4f}")
```

---

## 十一、聚类算法（Clustering）

### 11.1 K-Means聚类

**目标函数**：
$J = \sum_{i=1}^{k}\sum_{x\in C_i}||x - \mu_i||^2$

其中$\mu_i$是第$i$个簇的中心。

**算法步骤**：
1. 随机初始化K个聚类中心
2. 分配每个点到最近的中心
3.重新计算每个簇的中心
4. 重复2-3直到收敛

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 1. 生成聚类数据
from sklearn.datasets import make_blobs
X_cluster, y_true = make_blobs(n_samples=300, centers=4, 
                               cluster_std=0.60, random_state=42)

# 2. K-Means聚类
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X_cluster)

print(f"K-Means轮廓系数: {silhouette_score(X_cluster, y_kmeans):.4f}")
print(f"K-Means ARI: {adjusted_rand_score(y_true, y_kmeans):.4f}")

# 3. 肘部法则（选择最佳K值）
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster, kmeans.labels_))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, marker='o')
plt.xlabel('簇数量 K')
plt.ylabel('簇内平方和')
plt.title('肘部法则')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o')
plt.xlabel('簇数量 K')
plt.ylabel('轮廓系数')
plt.title('轮廓系数分析')
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=y_kmeans, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, edgecolors='black', label='中心点')
plt.title('K-Means聚类结果')
plt.legend()
plt.show()
```

### 11.2 DBSCAN（基于密度的聚类）

不需要指定簇数量，可以发现任意形状的簇。

**核心点**：邻域内至少有MinPts个点
**边界点**：不是核心点，但在核心点的邻域内
**噪声点**：既不是核心点也不是边界点

```python
# DBSCAN聚类
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X_cluster)

n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_noise = list(y_dbscan).count(-1)

print(f"DBSCAN发现的簇数量: {n_clusters}")
print(f"噪声点数量: {n_noise}")

plt.figure(figsize=(8, 6))
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=y_dbscan, cmap='viridis', s=50)
plt.title('DBSCAN聚类结果')
plt.show()
```

### 11.3 层次聚类（Hierarchical Clustering）

```python
# 层次聚类
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
y_hierarchical = hierarchical.fit_predict(X_cluster)

print(f"层次聚类轮廓系数: {silhouette_score(X_cluster, y_hierarchical):.4f}")

# 绘制树状图
from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(12, 6))
linked = linkage(X_cluster, method='ward')
dendrogram(linked)
plt.title('层次聚类树状图')
plt.xlabel('样本索引')
plt.ylabel('距离')
plt.show()
```

---

## 十二、降维算法（Dimensionality Reduction）

### 12.1 主成分分析（PCA）

将数据投影到方差最大的方向。

**目标**：找到投影方向$w$，最大化投影后的方差
$\max_w \frac{1}{n}\sum_{i=1}^{n}(w^Tx_i)^2$
$\text{s.t. } ||w|| = 1$

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# 1. 加载手写数字数据
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

print(f"原始数据维度: {X_digits.shape}")

# 2. PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_digits)

print(f"降维后维度: {X_pca.shape}")
print(f"方差解释率: {pca.explained_variance_ratio_}")
print(f"累积方差解释率: {pca.explained_variance_ratio_.sum():.4f}")

# 3. 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, 
                     cmap='tab10', alpha=0.5, s=5)
plt.colorbar(scatter, label='数字类别')
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.title('PCA降维可视化（手写数字）')
plt.show()

# 4. 选择最佳主成分数量
pca_full = PCA()
pca_full.fit(X_digits)

cumsum = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(cumsum)
plt.xlabel('主成分数量')
plt.ylabel('累积方差解释率')
plt.axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
plt.legend()
plt.grid(True)
plt.title('累积方差解释率')
plt.show()

# 找到解释95%方差的主成分数量
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"解释95%方差需要的主成分数量: {n_components_95}")
```

### 12.2 t-SNE（t-分布随机邻域嵌入）

非线性降维，特别适合可视化高维数据。

```python
from sklearn.manifold import TSNE

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_digits)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, 
                     cmap='tab10', alpha=0.5, s=5)
plt.colorbar(scatter, label='数字类别')
plt.title('t-SNE降维可视化')
plt.show()
```

### 12.3 线性判别分析（LDA）

有监督的降维方法，最大化类间距离，最小化类内距离。

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA降维（最多降到类别数-1维）
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_digits, y_digits)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_digits, 
                     cmap='tab10', alpha=0.5, s=5)
plt.colorbar(scatter, label='数字类别')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA降维可视化')
plt.show()

# 对比三种降维方法
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='tab10', alpha=0.5, s=5)
axes[0].set_title('PCA')

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10', alpha=0.5, s=5)
axes[1].set_title('t-SNE')

axes[2].scatter(X_lda[:, 0], X_lda[:, 1], c=y_digits, cmap='tab10', alpha=0.5, s=5)
axes[2].set_title('LDA')

plt.tight_layout()
plt.show()
```

---

## 十三、神经网络基础

### 13.1 感知机（Perceptron）

最简单的神经网络，单层线性分类器。

**激活函数**：
$y = \text{sign}(\sum_{i=1}^{n}w_ix_i + b)$

### 13.2 多层感知机（MLP）

**前向传播**：
$a^{[l]} = g^{[l]}(W^{[l]}a^{[l-1]} + b^{[l]})$

**常用激活函数**：
- **Sigmoid**：$\sigma(x) = \frac{1}{1+e^{-x}}$
- **Tanh**：$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- **ReLU**：$\text{ReLU}(x) = \max(0, x)$
- **Leaky ReLU**：$\text{LeakyReLU}(x) = \max(\alpha x, x)$

### 13.3 Python实现（使用sklearn）

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

# 1. 数据准备
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 创建MLP分类器
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 两个隐藏层
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# 3. 训练
mlp.fit(X_train_scaled, y_train)

# 4. 评估
train_score = mlp.score(X_train_scaled, y_train)
test_score = mlp.score(X_test_scaled, y_test)

print(f"训练集准确率: {train_score:.4f}")
print(f"测试集准确率: {test_score:.4f}")

# 5. 损失曲线
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_)
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('MLP训练损失曲线')
plt.grid(True)
plt.show()

# 6. 不同激活函数对比
activations = ['relu', 'tanh', 'logistic']
results = {}

for activation in activations:
    mlp_temp = MLPClassifier(hidden_layer_sizes=(50,), 
                             activation=activation,
                             max_iter=1000, 
                             random_state=42)
    mlp_temp.fit(X_train_scaled, y_train)
    results[activation] = mlp_temp.score(X_test_scaled, y_test)

plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.ylabel('测试集准确率')
plt.title('不同激活函数的性能对比')
plt.ylim(0.9, 1.0)
plt.grid(axis='y')
plt.show()
```

---

## 十四、模型评估与选择

### 14.1 评估指标

**分类问题**：
- **准确率（Accuracy）**：$\frac{TP + TN}{TP + TN + FP + FN}$
- **精确率（Precision）**：$\frac{TP}{TP + FP}$
- **召回率（Recall）**：$\frac{TP}{TP + FN}$
- **F1分数**：$F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$

**回归问题**：
- **MAE**：$\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- **MSE**：$\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **RMSE**：$\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$
- **R²**：$1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 分类评估示例
y_pred = mlp.predict(X_test_scaled)
y_pred_proba = mlp.predict_proba(X_test_scaled)

print("=== 分类评估指标 ===")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"宏平均精确率: {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"宏平均召回率: {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"宏平均F1: {f1_score(y_test, y_pred, average='macro'):.4f}")

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('混淆矩阵')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.ylabel('真实标签')
plt.xlabel('预测标签')

# 在格子中显示数值
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()
```

### 14.2 交叉验证

```python
from sklearn.model_selection import cross_val_score, cross_validate, KFold

# 1. K折交叉验证
cv_scores = cross_val_score(mlp, X_train_scaled, y_train, cv=5)
print(f"5折交叉验证分数: {cv_scores}")
print(f"平均分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 2. 分层K折交叉验证（保持类别比例）
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_scores = cross_val_score(mlp, X_train_scaled, y_train, cv=skf)
print(f"分层5折交叉验证分数: {skf_scores.mean():.4f}")

# 3. 多指标交叉验证
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(mlp, X_train_scaled, y_train, 
                           cv=5, scoring=scoring, return_train_score=True)

for metric in scoring:
    print(f"{metric}:")
    print(f"  训练: {cv_results[f'train_{metric}'].mean():.4f}")
    print(f"  验证: {cv_results[f'test_{metric}'].mean():.4f}")
```

### 14.3 学习曲线与验证曲线

```python
from sklearn.model_selection import learning_curve, validation_curve

# 1. 学习曲线
train_sizes, train_scores, val_scores = learning_curve(
    MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    X_train_scaled, y_train,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='训练分数', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, 
                train_mean + train_std, alpha=0.15)
plt.plot(train_sizes, val_mean, label='验证分数', marker='s')
plt.fill_between(train_sizes, val_mean - val_std, 
                val_mean + val_std, alpha=0.15)
plt.xlabel('训练样本数')
plt.ylabel('分数')
plt.title('学习曲线')
plt.legend()
plt.grid(True)
plt.show()

# 2. 验证曲线（超参数调优）
param_range = [10, 20, 50, 100, 200]
train_scores, val_scores = validation_curve(
    MLPClassifier(max_iter=1000, random_state=42),
    X_train_scaled, y_train,
    param_name='hidden_layer_sizes',
    param_range=[(n,) for n in param_range],
    cv=5
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label='训练分数', marker='o')
plt.plot(param_range, val_mean, label='验证分数', marker='s')
plt.xlabel('隐藏层神经元数量')
plt.ylabel('分数')
plt.title('验证曲线')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.show()
```

### 14.4 超参数调优

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# 1. 网格搜索
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

grid_search = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print("最佳参数:")
print(grid_search.best_params_)
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
print(f"测试集分数: {grid_search.score(X_test_scaled, y_test):.4f}")

# 2. 随机搜索（更高效）
param_distributions = {
    'hidden_layer_sizes': [(randint(10, 200).rvs(),) for _ in range(10)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': uniform(0.0001, 0.01),
    'learning_rate': ['constant', 'adaptive']
}

random_search = RandomizedSearchCV(
    MLPClassifier(max_iter=1000, random_state=42),
    param_distributions,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)

print("\n随机搜索最佳参数:")
print(random_search.best_params_)
print(f"最佳交叉验证分数: {random_search.best_score_:.4f}")
```

---

## 十五、特征工程

### 15.1 特征缩放

```python
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, 
                                   RobustScaler, Normalizer)

# 生成示例数据
X_example = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [100, 200]])

# 1. 标准化（Z-score）
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X_example)
print("标准化后:")
print(X_standard)

# 2. 归一化到[0,1]
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X_example)
print("\n归一化后:")
print(X_minmax)

# 3. 鲁棒缩放（对异常值不敏感）
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X_example)
print("\n鲁棒缩放后:")
print(X_robust)

# 4. L2归一化
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X_example)
print("\nL2归一化后:")
print(X_normalized)
```

### 15.2 特征编码

```python
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, 
                                   OrdinalEncoder)
import pandas as pd

# 示例数据
data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue', 'red'],
    'size': ['S', 'M', 'L', 'M', 'L'],
    'price': [10, 15, 20, 15, 22]
})

# 1. 标签编码（用于目标变量）
le = LabelEncoder()
data['color_encoded'] = le.fit_transform(data['color'])
print("标签编码:")
print(data[['color', 'color_encoded']])

# 2. 独热编码（用于名义变量）
ohe = OneHotEncoder(sparse_output=False)
color_ohe = ohe.fit_transform(data[['color']])
color_ohe_df = pd.DataFrame(color_ohe, columns=ohe.get_feature_names_out(['color']))
print("\n独热编码:")
print(color_ohe_df)

# 3. 序数编码（用于有序变量）
size_order = ['S', 'M', 'L']
oe = OrdinalEncoder(categories=[size_order])
data['size_encoded'] = oe.fit_transform(data[['size']])
print("\n序数编码:")
print(data[['size', 'size_encoded']])
```

### 15.3 特征选择

```python
from sklearn.feature_selection import (SelectKBest, chi2, f_classif,
                                       RFE, SelectFromModel)

# 使用鸢尾花数据
X_iris, y_iris = load_iris(return_X_y=True)
feature_names = load_iris().feature_names

# 1. 单变量特征选择
selector_chi2 = SelectKBest(chi2, k=2)
X_selected_chi2 = selector_chi2.fit_transform(X_iris, y_iris)

print("卡方检验选择的特征:")
selected_features = [feature_names[i] for i in selector_chi2.get_support(indices=True)]
print(selected_features)
print(f"特征分数: {selector_chi2.scores_}")

# 2. 递归特征消除（RFE）
estimator = LogisticRegression(max_iter=1000)
rfe = RFE(estimator, n_features_to_select=2)
X_selected_rfe = rfe.fit_transform(X_iris, y_iris)

print("\nRFE选择的特征:")
selected_features_rfe = [feature_names[i] for i in rfe.get_support(indices=True)]
print(selected_features_rfe)

# 3. 基于模型的特征选择
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
selector_model = SelectFromModel(rf_selector, threshold='median')
X_selected_model = selector_model.fit_transform(X_iris, y_iris)

print("\n基于随机森林的特征选择:")
selected_features_model = [feature_names[i] for i in selector_model.get_support(indices=True)]
print(selected_features_model)

# 可视化特征重要性
rf_selector.fit(X_iris, y_iris)
importances = rf_selector.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.title('特征重要性')
plt.tight_layout()
plt.show()
```

### 15.4 多项式特征

```python
from sklearn.preprocessing import PolynomialFeatures

# 生成非线性数据
X_poly = np.array([[1, 2], [2, 3], [3, 4]])

# 创建多项式特征（degree=2）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_transformed = poly.fit_transform(X_poly)

print("原始特征:")
print(X_poly)
print(f"\n多项式特征（degree=2）:")
print(X_poly_transformed)
print(f"特征名称: {poly.get_feature_names_out(['x1', 'x2'])}")

# 多项式回归示例
X_poly_reg = np.sort(5 * np.random.rand(40, 1), axis=0)
y_poly_reg = np.sin(X_poly_reg).ravel() + np.random.randn(40) * 0.1

degrees = [1, 3, 10]
plt.figure(figsize=(15, 4))

for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_train = poly_features.fit_transform(X_poly_reg)
    
    model = LinearRegression()
    model.fit(X_poly_train, y_poly_reg)
    
    X_test_plot = np.linspace(0, 5, 100).reshape(-1, 1)
    X_test_poly = poly_features.transform(X_test_plot)
    y_pred_plot = model.predict(X_test_poly)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(X_poly_reg, y_poly_reg, s=20, edgecolor="black", c="darkorange")
    plt.plot(X_test_plot, y_pred_plot, color="cornflowerblue", linewidth=2)
    plt.title(f'多项式回归 (degree={degree})')
    plt.xlabel('X')
    plt.ylabel('y')

plt.tight_layout()
plt.show()
```

---

## 十六、处理不平衡数据

### 16.1 问题识别

```python
from sklearn.datasets import make_classification
from collections import Counter

# 创建不平衡数据集
X_imb, y_imb = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, n_classes=2, weights=[0.9, 0.1],
    random_state=42
)

print(f"类别分布: {Counter(y_imb)}")
print(f"不平衡比例: {Counter(y_imb)[0] / Counter(y_imb)[1]:.2f}:1")
```

### 16.2 处理方法

```python
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42
)

# 1. 随机过采样
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train_imb, y_train_imb)
print(f"随机过采样后: {Counter(y_ros)}")

# 2. SMOTE（合成少数类过采样）
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_imb, y_train_imb)
print(f"SMOTE后: {Counter(y_smote)}")

# 3. 随机欠采样
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train_imb, y_train_imb)
print(f"随机欠采样后: {Counter(y_rus)}")

# 4. SMOTE + Tomek Links组合
smote_tomek = SMOTETomek(random_state=42)
X_combined, y_combined = smote_tomek.fit_resample(X_train_imb, y_train_imb)
print(f"SMOTE + Tomek后: {Counter(y_combined)}")

# 5. 类别权重
clf_weighted = LogisticRegression(class_weight='balanced', max_iter=1000)
clf_weighted.fit(X_train_imb, y_train_imb)

# 6. 性能对比
methods = {
    '原始数据': (X_train_imb, y_train_imb),
    '随机过采样': (X_ros, y_ros),
    'SMOTE': (X_smote, y_smote),
    '随机欠采样': (X_rus, y_rus),
    'SMOTE+Tomek': (X_combined, y_combined)
}

results_imbalance = {}
for name, (X_train_method, y_train_method) in methods.items():
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_method, y_train_method)
    y_pred = clf.predict(X_test_imb)
    
    results_imbalance[name] = {
        'accuracy': accuracy_score(y_test_imb, y_pred),
        'precision': precision_score(y_test_imb, y_pred),
        'recall': recall_score(y_test_imb, y_pred),
        'f1': f1_score(y_test_imb, y_pred)
    }

# 可视化对比
metrics = ['accuracy', 'precision', 'recall', 'f1']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, metric in enumerate(metrics):
    values = [results_imbalance[method][metric] for method in methods.keys()]
    axes[i].bar(methods.keys(), values)
    axes[i].set_ylabel(metric.capitalize())
    axes[i].set_xticklabels(methods.keys(), rotation=45, ha='right')
    axes[i].set_ylim(0, 1)
    axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 十七、时间序列预测基础

### 17.1 移动平均与指数平滑

```python
import pandas as pd
from datetime import datetime, timedelta

# 生成时间序列数据
dates = pd.date_range('2020-01-01', periods=365, freq='D')
np.random.seed(42)
trend = np.linspace(100, 200, 365)
seasonal = 20 * np.sin(np.linspace(0, 4*np.pi, 365))
noise = np.random.randn(365) * 5
values = trend + seasonal + noise

ts_data = pd.DataFrame({'date': dates, 'value': values})
ts_data.set_index('date', inplace=True)

# 1. 移动平均
ts_data['MA_7'] = ts_data['value'].rolling(window=7).mean()
ts_data['MA_30'] = ts_data['value'].rolling(window=30).mean()

# 2. 指数加权移动平均
ts_data['EWMA'] = ts_data['value'].ewm(span=30).mean()

# 可视化
plt.figure(figsize=(14, 6))
plt.plot(ts_data.index, ts_data['value'], label='原始数据', alpha=0.5)
plt.plot(ts_data.index, ts_data['MA_7'], label='7天移动平均')
plt.plot(ts_data.index, ts_data['MA_30'], label='30天移动平均')
plt.plot(ts_data.index, ts_data['EWMA'], label='指数加权移动平均')
plt.xlabel('日期')
plt.ylabel('值')
plt.title('时间序列平滑')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

### 17.2 ARIMA模型

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 准备数据
train_size = int(len(ts_data) * 0.8)
train, test = ts_data['value'][:train_size], ts_data['value'][train_size:]

# 1. ACF和PACF图（帮助选择p, q参数）
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(train, lags=30, ax=axes[0])
plot_pacf(train, lags=30, ax=axes[1])
axes[0].set_title('自相关函数(ACF)')
axes[1].set_title('偏自相关函数(PACF)')
plt.tight_layout()
plt.show()

# 2. 训练ARIMA模型
model = ARIMA(train, order=(1, 1, 1))  # (p, d, q)
fitted_model = model.fit()

print(fitted_model.summary())

# 3. 预测
forecast_steps = len(test)
forecast = fitted_model.forecast(steps=forecast_steps)

# 4. 可视化预测结果
plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label='训练数据')
plt.plot(test.index, test, label='测试数据', color='orange')
plt.plot(test.index, forecast, label='ARIMA预测', color='red')
plt.xlabel('日期')
plt.ylabel('值')
plt.title('ARIMA时间序列预测')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. 评估
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"\nMAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
```

---

## 十八、推荐系统基础

### 18.1 协同过滤

```python
from sklearn.metrics.pairwise import cosine_similarity

# 创建用户-物品评分矩阵
ratings = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 0, 1, 3],
    [1, 1, 0, 5, 2],
    [1, 0, 0, 4, 0],
    [0, 1, 5, 4, 0],
])

users = ['用户A', '用户B', '用户C', '用户D', '用户E']
items = ['物品1', '物品2', '物品3', '物品4', '物品5']

# 1. 基于用户的协同过滤
user_similarity = cosine_similarity(ratings)

print("用户相似度矩阵:")
user_sim_df = pd.DataFrame(user_similarity, index=users, columns=users)
print(user_sim_df)

# 预测用户A对物品3的评分
target_user = 0
target_item = 2

# 找到与目标用户相似的用户
similar_users = user_similarity[target_user]
# 获取这些用户对目标物品的评分
item_ratings = ratings[:, target_item]
# 计算加权平均
mask = item_ratings > 0  # 只考虑有评分的用户
if mask.sum() > 0:
    predicted_rating = np.sum(similar_users[mask] * item_ratings[mask]) / np.sum(similar_users[mask])
    print(f"\n预测{users[target_user]}对{items[target_item]}的评分: {predicted_rating:.2f}")

# 2. 基于物品的协同过滤
item_similarity = cosine_similarity(ratings.T)

print("\n物品相似度矩阵:")
item_sim_df = pd.DataFrame(item_similarity, index=items, columns=items)
print(item_sim_df)

# 可视化相似度矩阵
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(user_similarity, cmap='YlOrRd')
axes[0].set_xticks(range(len(users)))
axes[0].set_yticks(range(len(users)))
axes[0].set_xticklabels(users, rotation=45)
axes[0].set_yticklabels(users)
axes[0].set_title('用户相似度')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(item_similarity, cmap='YlOrRd')
axes[1].set_xticks(range(len(items)))
axes[1].set_yticks(range(len(items)))
axes[1].set_xticklabels(items, rotation=45)
axes[1].set_yticklabels(items)
axes[1].set_title('物品相似度')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()
```

### 18.2 矩阵分解（SVD）

```python
from scipy.sparse.linalg import svds

# 对评分矩阵进行SVD分解
U, sigma, Vt = svds(ratings, k=3)
sigma = np.diag(sigma)

# 重构评分矩阵
predicted_ratings = np.dot(np.dot(U, sigma), Vt)

print("原始评分矩阵:")
print(ratings)
print("\n预测评分矩阵:")
print(predicted_ratings.round(2))

# 为用户A推荐物品
user_idx = 0
user_ratings = ratings[user_idx]
user_predictions = predicted_ratings[user_idx]

# 找出用户未评分的物品
unrated_items = np.where(user_ratings == 0)[0]
recommendations = sorted(zip(unrated_items, user_predictions[unrated_items]), 
                        key=lambda x: x[1], reverse=True)

print(f"\n为{users[user_idx]}推荐:")
for item_idx, score in recommendations:
    print(f"  {items[item_idx]}: {score:.2f}")
```

---

## 十九、异常检测

### 19.1 孤立森林（Isolation Forest）

```python
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

# 生成正常数据和异常数据
np.random.seed(42)
X_normal = 0.3 * np.random.randn(200, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X_anomaly = np.r_[X_normal, X_outliers]
y_true = np.array([0] * 200 + [1] * 20)

# 1. 孤立森林
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred_iso = iso_forest.fit_predict(X_anomaly)
y_pred_iso = (y_pred_iso == -1).astype(int)

# 2. 椭圆包络
elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
y_pred_elliptic = elliptic.fit_predict(X_anomaly)
y_pred_elliptic = (y_pred_elliptic == -1).astype(int)

# 3. One-Class SVM
oc_svm = OneClassSVM(nu=0.1, gamma='auto')
y_pred_svm = oc_svm.fit_predict(X_anomaly)
y_pred_svm = (y_pred_svm == -1).astype(int)

# 可视化
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

# 真实情况
axes[0].scatter(X_normal[:, 0], X_normal[:, 1], c='blue', s=20, label='正常')
axes[0].scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=50, 
               marker='x', label='异常')
axes[0].set_title('真实标签')
axes[0].legend()

# 孤立森林
axes[1].scatter(X_anomaly[y_pred_iso==0][:, 0], X_anomaly[y_pred_iso==0][:, 1], 
               c='blue', s=20, label='正常')
axes[1].scatter(X_anomaly[y_pred_iso==1][:, 0], X_anomaly[y_pred_iso==1][:, 1], 
               c='red', s=50, marker='x', label='异常')
axes[1].set_title(f'孤立森林 (准确率: {accuracy_score(y_true, y_pred_iso):.2f})')
axes[1].legend()

# 椭圆包络
axes[2].scatter(X_anomaly[y_pred_elliptic==0][:, 0], 
               X_anomaly[y_pred_elliptic==0][:, 1], 
               c='blue', s=20, label='正常')
axes[2].scatter(X_anomaly[y_pred_elliptic==1][:, 0], 
               X_anomaly[y_pred_elliptic==1][:, 1], 
               c='red', s=50, marker='x', label='异常')
axes[2].set_title(f'椭圆包络 (准确率: {accuracy_score(y_true, y_pred_elliptic):.2f})')
axes[2].legend()

# One-Class SVM
axes[3].scatter(X_anomaly[y_pred_svm==0][:, 0], X_anomaly[y_pred_svm==0][:, 1], 
               c='blue', s=20, label='正常')
axes[3].scatter(X_anomaly[y_pred_svm==1][:, 0], X_anomaly[y_pred_svm==1][:, 1], 
               c='red', s=50, marker='x', label='异常')
axes[3].set_title(f'One-Class SVM (准确率: {accuracy_score(y_true, y_pred_svm):.2f})')
axes[3].legend()

plt.tight_layout()
plt.show()
```

---

## 二十、模型解释与可解释性

### 20.1 SHAP值

```python
import shap

# 训练模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 计算SHAP值
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# 单个预测解释
shap.initjs()
idx = 0
shap.force_plot(explainer.expected_value[1], shap_values[1][idx], 
                X_test[idx], feature_names=iris.feature_names)

# 特征重要性总结
shap.summary_plot(shap_values[1], X_test, feature_names=iris.feature_names)
```

### 20.2 部分依赖图（Partial Dependence Plot）

```python
from sklearn.inspection import PartialDependenceDisplay

# 绘制部分依赖图
fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(
    rf_model, X_train, features=[0, 1, 2, 3],
    feature_names=iris.feature_names,
    ax=ax
)
plt.tight_layout()
plt.show()
```

---

## 二十一、实战建议

### 21.1 机器学习项目流程

```
1. 问题定义
   ├─ 明确业务目标
   ├─ 确定评估指标
   └─ 定义成功标准

2. 数据收集与探索
   ├─ 收集数据
   ├─ 探索性数据分析（EDA）
   ├─ 数据可视化
   └─ 识别数据质量问题

3. 数据预处理
   ├─ 处理缺失值
   ├─ 处理异常值
   ├─ 特征缩放
   ├─ 特征编码
   └─ 特征工程

4. 模型选择与训练
   ├─ 建立基线模型
   ├─ 尝试多种算法
   ├─ 交叉验证
   └─ 超参数调优

5. 模型评估
   ├─ 多指标评估
   ├─ 学习曲线分析
   ├─ 错误分析
   └─ 模型解释

6. 模型部署
   ├─ 模型保存
   ├─ API开发
   ├─ 监控和维护
   └─ 持续改进
```

### 21.2 常见陷阱与解决方案

| 陷阱 | 原因 | 解决方案 |
|------|------|----------|
| 数据泄漏 | 测试集信息泄露到训练集 | 严格划分数据集，使用Pipeline |
| 过拟合 | 模型过于复杂 | 正则化、交叉验证、更多数据 |
| 欠拟合 | 模型过于简单 | 增加模型复杂度、特征工程 |
| 样本不平衡 | 类别分布不均 | 重采样、类别权重、集成方法 |
| 特征尺度差异 | 不同特征量纲不同 | 特征标准化/归一化 |
| 过度调参 | 在测试集上反复调参 | 使用验证集、交叉验证 |
| 忽视基线 | 没有简单模型对比 | 先建立简单基线模型 |
| 评估指标单一 | 只看准确率 | 使用多个评估指标 |

### 21.3 模型保存与加载

```python
import joblib
import pickle

# 1. 使用joblib（推荐，适合大型numpy数组）
joblib.dump(rf_model, 'random_forest_model.pkl')
loaded_model = joblib.load('random_forest_model.pkl')

# 2. 使用pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('model.pkl', 'rb') as f:
    loaded_model_pickle = pickle.load(f)

# 验证加载的模型
print(f"原模型准确率: {rf_model.score(X_test, y_test):.4f}")
print(f"加载模型准确率: {loaded_model.score(X_test, y_test):.4f}")
```

### 21.4 Pipeline构建

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 完整的机器学习Pipeline
numerical_features = [0, 1, 2, 3]
categorical_features = []

# 数值特征处理
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)
    ])

# 完整Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 训练
pipeline.fit(X_train, y_train)

# 预测
y_pred_pipeline = pipeline.predict(X_test)
print(f"Pipeline准确率: {accuracy_score(y_test, y_pred_pipeline):.4f}")

# Pipeline可以直接保存
joblib.dump(pipeline, 'complete_pipeline.pkl')
```

---

## 二十二、学习路径建议

### 初级阶段（1-3个月）
1. ✅ 掌握Python基础和NumPy、Pandas
2. ✅ 理解基本概念：特征、标签、损失函数
3. ✅ 学习线性回归、逻辑回归
4. ✅ 熟悉Scikit-learn基本用法
5. ✅ 完成简单项目（鸢尾花分类、房价预测）

### 中级阶段（3-6个月）
1. ✅ 深入理解决策树、SVM、集成学习
2. ✅ 掌握特征工程技巧
3. ✅ 学习模型评估和调优
4. ✅ 理解过拟合和欠拟合
5. ✅ 参与Kaggle入门竞赛

### 高级阶段（6-12个月）
1. ✅ 学习神经网络和深度学习
2. ✅ 掌握时间序列、推荐系统等专题
3. ✅ 理解模型可解释性
4. ✅ 学习模型部署和生产化
5. ✅ 完成端到端项目

---

## 二十三、推荐资源

### 📚 书籍
- **入门**：《Python机器学习基础教程》
- **进阶**：《机器学习》周志华（西瓜书）
- **深入**：《统计学习方法》李航
- **实战**：《机器学习实战》Peter Harrington

### 🎓 在线课程
- Andrew Ng的《Machine Learning》
- Fast.ai的《Practical Deep Learning》
- Coursera的《机器学习专项课程》

### 🛠️ 工具库
- **核心库**：Scikit-learn、NumPy、Pandas
- **可视化**：Matplotlib、Seaborn、Plotly
- **深度学习**：TensorFlow、PyTorch、Keras
- **不平衡数据**：imbalanced-learn
- **自动化ML**：Auto-sklearn、H2O.ai

### 🏆 实践平台
- **Kaggle**：竞赛和数据集
- **天池**：阿里云机器学习竞赛
- **Google Colab**：免费GPU
- **UCI ML Repository**：经典数据集

---

## 总结

机器学习是一个广阔而深入的领域，本笔记涵盖了：

✅ **监督学习**：线性回归、逻辑回归、决策树、SVM、KNN、朴素贝叶斯、神经网络
✅ **无监督学习**：K-Means、DBSCAN、层次聚类、PCA、t-SNE、LDA
✅ **集成学习**：随机森林、AdaBoost、Gradient Boosting、XGBoost、Stacking
✅ **评估与优化**：交叉验证、网格搜索、学习曲线、超参数调优
✅ **特征工程**：特征缩放、编码、选择、多项式特征
✅ **专题应用**：不平衡数据、时间序列、推荐系统、异常检测
✅ **实战技巧**：Pipeline、模型保存、项目流程

**核心理念**：
- 从简单模型开始，逐步增加复杂度
- 数据质量比算法更重要
- 理解原理比记忆公式更重要
- 多实践、多思考、多总结

**持续学习**：
机器学习发展迅速，保持学习热情，关注最新研究和技术动态！

---

*Happy Machine Learning! 🚀*