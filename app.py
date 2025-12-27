from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# 加载预训练模型和编码器
model = joblib.load('diet_model.pkl')
le_gender = joblib.load('le_gender.pkl')
le_goal = joblib.load('le_goal.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # 获取输入
    gender = data['gender']  # '男' or '女'
    age = data['age']
    height = data['height']
    weight = data['weight']
    goal = data['goal']  # '减脂', '增肌', '维持'

    # 编码
    gender_encoded = le_gender.transform([gender])[0]
    goal_encoded = le_goal.transform([goal])[0]

    # 标准化
    features = np.array([[gender_encoded, age, height, weight, goal_encoded]])
    features_scaled = scaler.transform(features)

    # 预测
    prediction = model.predict(features_scaled)[0]

    return jsonify({
        'calories': float(prediction),
        'message': f'建议每日摄入 {prediction:.0f} 千卡'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)