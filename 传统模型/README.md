# 能源数据预测模型

这个项目实现了一个基于SARIMAX模型的能源数据预测系统。该系统可以处理时间序列数据，进行特征工程，训练预测模型，并生成可视化结果。

## 功能特点

- 数据预处理和特征工程
- SARIMAX模型训练和参数优化
- 预测结果可视化
- 支持中英文双语输出

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据：
   - 确保数据为CSV格式
   - 数据应包含时间戳列和数值列

2. 运行预测：
   ```python
   from energy_forecast import train_and_predict
   
   # 训练模型并生成预测
   results = train_and_predict('your_data.csv')
   ```

3. 查看结果：
   - 预测结果将保存在 `results` 目录下
   - 可视化图表将自动保存为PNG格式

## 注意事项

- 确保数据质量，处理缺失值和异常值
- 根据实际需求调整模型参数
- 定期更新模型以适应新的数据模式 