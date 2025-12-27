# generate_data.py
import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000
data = []

for _ in range(n):
    gender = np.random.choice(['男', '女'])
    if gender == '男':
        height = np.random.normal(175, 8)
        weight = np.random.normal(70, 12)
    else:
        height = np.random.normal(162, 7)
        weight = np.random.normal(58, 10)

    age = np.random.randint(18, 60)
    goal = np.random.choice(['减脂', '增肌', '保持'], p=[0.4, 0.3, 0.3])
    activity = np.random.choice(['低', '中等', '高'], p=[0.2, 0.5, 0.3])

    data.append([round(height, 1), round(weight, 1), age, gender, goal, activity])

# 创建 DataFrame 并保存为 CSV
df = pd.DataFrame(data, columns=['height', 'weight', 'age', 'gender', 'goal', 'activity'])
df.to_csv('users.csv', index=False, encoding='utf-8-sig')
print("✅ 成功生成 users.csv，共", len(df), "条数据")