# predict.py
import argparse
import json
import numpy as np
import pickle

# 加载模型和 scaler（使用 encoding='latin1' 解决兼容性问题）
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f, encoding='latin1')

with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f, encoding='latin1')

# 加载标签编码器
with open('le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f, encoding='latin1')
with open('le_goal.pkl', 'rb') as f:
    le_goal = pickle.load(f, encoding='latin1')
with open('le_activity.pkl', 'rb') as f:
    le_activity = pickle.load(f, encoding='latin1')

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--height', type=float, required=True)
parser.add_argument('--weight', type=float, required=True)
parser.add_argument('--age', type=int, required=True)
parser.add_argument('--gender', type=str, required=True)
parser.add_argument('--goal', type=str, required=True)
parser.add_argument('--activity', type=str, required=True)
args = parser.parse_args()

# 对分类变量进行编码
try:
    gender_encoded = le_gender.transform([args.gender])[0]
    goal_encoded = le_goal.transform([args.goal])[0]
    activity_encoded = le_activity.transform([args.activity])[0]
except ValueError as e:
    print(json.dumps({"error": f"输入值不在训练集中: {e}"}))
    exit(1)

# 构造输入向量
X = np.array([[args.height, args.weight, args.age, gender_encoded, goal_encoded, activity_encoded]])

# 标准化
X_scaled = scaler.transform(X)

# 预测聚类
cluster_id = kmeans.predict(X_scaled)[0]

# 输出结果为 JSON（确保最后一行是纯 JSON）
print(json.dumps({"clusterId": int(cluster_id)}))