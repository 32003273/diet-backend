// server.js
const express = require('express');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000; // ← 关键：使用云平台分配的端口

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  next();
});

app.get('/predict', (req, res) => {
  const { height, weight, age, gender, goal, activity } = req.query;

  if (!height || !weight || !age || !gender || !goal || !activity) {
    return res.status(400).json({ error: '缺少参数' });
  }

  const pythonScript = path.join(__dirname, 'predict.py');
  const cmd = `python "${pythonScript}" --height ${height} --weight ${weight} --age ${age} --gender "${gender}" --goal "${goal}" --activity "${activity}"`;

  exec(cmd, { encoding: 'utf8' }, (error, stdout, stderr) => {
    if (error) {
      return res.status(500).json({ error: '模型出错', details: stderr });
    }

    try {
      const lines = stdout.trim().split('\n');
      let jsonStr = lines.find(line => line.includes('{')) || '';
      const result = JSON.parse(jsonStr);
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: '解析失败', output: stdout });
    }
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ 服务器运行在端口 ${PORT}`);
});