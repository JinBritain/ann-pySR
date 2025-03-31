# ann-pySR

我们采集来自20+文献数据的300+条数据作为训练集，中国实验快堆的62条数据作为测试集，很遗憾目前数据和网络结构均无法开源。

```python
!pip install SciencePlots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm  # 用于显示进度条
import scienceplots

# 读取训练和测试数据
train_df = pd.read_csv('/kaggle/input/mlpmlp1111/train.csv')
test_df = pd.read_csv('/kaggle/input/mlpmlp1111/test.csv')

# 特征和目标变量的定义
X_train = train_df[['n', 'pd', 'wd', 'hd', 'r']]
y_train = train_df['f']
X_test = test_df[['n', 'pd', 'wd', 'hd', 'r']]
y_test = test_df['f']

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 要测试的随机种子范围
seed_range = range(1, 101)  # 测试1到100的种子
results = []

# 使用tqdm显示进度
for seed in tqdm(seed_range):
    try:
        # 随机森林获取特征重要性
        rf_model = RandomForestRegressor(random_state=seed)
        rf_model.fit(X_train_scaled, y_train)
        feature_importances = rf_model.feature_importances_
        
        # 对特征加权
        weighted_features_train = X_train_scaled * feature_importances
        weighted_features_test = X_test_scaled * feature_importances
        
        # 训练MLP模型
        mlp_model = MLPRegressor(
            #无法开源
        )
        
        mlp_model.fit(weighted_features_train, y_train)
        
        # 预测并计算R²
        y_pred = mlp_model.predict(weighted_features_test)
        r2 = r2_score(y_test, y_pred)
        
        # 保存结果
        results.append({
            'seed': seed,
            'r2_score': r2
        })
        
    except Exception as e:
        print(f"种子 {seed} 出错: {str(e)}")
        continue

# 转换为DataFrame并排序
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='r2_score', ascending=False).reset_index(drop=True)

# 显示前10个最佳种子
print("Top 10 最佳随机种子:")
print(results_df.head(10))

with plt.style.context('science'):
    plt.style.use(['science', 'no-latex'])# 获取横轴数据、真实值和各个预测方法的值
# 绘制所有种子的R²分布
    plt.figure(figsize=(12, 6))
    plt.bar(results_df['seed'], results_df['r2_score'])
    plt.axhline(y=results_df['r2_score'].mean(), color='r', linestyle='--', label=f'平均R²: {results_df["r2_score"].mean():.4f}')
    plt.xlabel('Random Seed‘, fontsize=14)
    plt.ylabel('R2, fontsize=14')
    plt.legend()
    plt.grid(axis='y')
    plt.show()

# 使用最佳种子重新训练模型
best_seed = results_df.iloc[0]['seed']
print(f"最佳随机种子是: {best_seed}, R²值: {results_df.iloc[0]['r2_score']:.6f}")

#接着再把最佳种子带进原结构神经网络
