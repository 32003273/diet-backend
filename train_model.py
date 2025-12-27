# train_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import pickle

# 1. 加载数据（使用 read_excel 因为实际是 Excel 文件）
df = pd.read_excel('users.xlsx')

# 2. 数据预处理
# 假设列名为: height, weight, age, gender, goal, activity
features = ['height', 'weight', 'age', 'gender', 'goal', 'activity']
X = df[features].copy()

# 对分类变量进行标签编码
le_gender = LabelEncoder()
le_goal = LabelEncoder()
le_activity = LabelEncoder()

X['gender'] = le_gender.fit_transform(X['gender'])
X['goal'] = le_goal.fit_transform(X['goal'])
X['activity'] = le_activity.fit_transform(X['activity'])

# 3. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 训练 KMeans（聚成 3 类）
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# 5. 保存模型和编码器（使用 protocol=4 确保兼容性）
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f, protocol=4)

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f, protocol=4)

with open('le_gender.pkl', 'wb') as f:
    pickle.dump(le_gender, f, protocol=4)

with open('le_goal.pkl', 'wb') as f:
    pickle.dump(le_goal, f, protocol=4)

with open('le_activity.pkl', 'wb') as f:
    pickle.dump(le_activity, f, protocol=4)

print("✅ 模型已成功训练并保存为兼容格式！")
print("生成文件：scaler.pkl, kmeans_model.pkl, le_*.pkl")