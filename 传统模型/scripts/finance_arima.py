import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller
import os
import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 
                                  'WenQuanYi Micro Hei', 'PingFang SC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建保存结果的文件夹
if not os.path.exists('finance_reports'):
    os.makedirs('finance_reports')

def check_stationarity(data, title):
    """
    检查时间序列的平稳性
    """
    # 执行ADF测试
    result = adfuller(data)
    
    print(f"\n{title}的ADF检验结果:")
    print(f'ADF统计量: {result[0]:.4f}')
    print(f'p值: {result[1]:.4f}')
    print('临界值:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    # 绘制时间序列图
    plt.figure(figsize=(15, 5))
    plt.plot(data.index, data.values)
    plt.title(f'{title}时间序列图')
    plt.xlabel('日期')
    plt.ylabel('值')
    plt.grid(True)
    plt.savefig(f'finance_reports/{title}_timeseries.png')
    plt.close()
    
    return result[1] < 0.05  # 返回是否平稳（p值<0.05为平稳）

# 读取数据
df = pd.read_csv('GOOGL_standardized.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 检查原始收盘价的平稳性
print("\n数据预处理...")
print(f"原始数据长度: {len(df)}")
is_stationary = check_stationarity(df['Close'], '原始收盘价')

if not is_stationary:
    print("\n原始数据不平稳，进行对数转换和差分处理...")
    
    # 对数转换
    df['Log_Close'] = np.log(df['Close'])
    is_log_stationary = check_stationarity(df['Log_Close'], '对数转换后的收盘价')
    
    # 一阶差分
    df['Diff_Log_Close'] = df['Log_Close'].diff()
    # 删除差分后产生的NaN值
    df = df.dropna()
    print(f"差分后数据长度: {len(df)}")
    is_diff_stationary = check_stationarity(df['Diff_Log_Close'], '对数差分后的收盘价')
    
    # 使用处理后的数据
    target_data = df['Diff_Log_Close']
    print("\n使用对数差分后的数据进行建模...")
else:
    target_data = df['Close']
    print("\n原始数据已经平稳，直接使用原始数据建模...")

# 划分训练集和测试集（确保长度一致）
n_test = 40  # 测试集大小
train_size = len(target_data) - n_test
train_data = target_data[:train_size]
test_data = target_data[train_size:train_size + n_test]

print(f"\n数据集大小:")
print(f"总数据量: {len(target_data)}")
print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")

# 保存用于后续转换的原始数据索引
original_train_index = df.index[:train_size]
original_test_index = df.index[train_size:train_size + n_test]

print(f"测试集索引长度: {len(original_test_index)}")

def calculate_technical_indicators(df):
    # 价格变化
    df['Returns'] = df['Close'].pct_change()
    
    # 波动率
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # 移动平均
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 趋势指标
    df['Price_MA5_Ratio'] = df['Close'] / df['MA5']
    df['Price_MA20_Ratio'] = df['Close'] / df['MA20']
    
    # 成交量指标
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    
    return df

# 准备数据
df = calculate_technical_indicators(df)
df = df.dropna()

# 选择特征
features = ['Close', 'Returns', 'Volatility', 'Price_MA5_Ratio', 'Price_MA20_Ratio', 'Volume_Ratio']

# 数据标准化
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[features]), 
    columns=features,
    index=df.index
)

# 设置SARIMAX模型参数
print("\n使用预设的SARIMAX参数...")
# 调整参数以避免冲突
order = (4, 1, 4)
seasonal_order = (1, 1, 1, 10)

print(f"\nSARIMAX模型参数:")
print(f"非季节性参数 (p,d,q): {order}")
print(f"季节性参数 (P,D,Q,s): {seasonal_order}")

# 训练SARIMAX模型
print("\n训练SARIMAX模型...")
model = SARIMAX(train_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit(disp=False)

# 预测
print("\n生成预测结果...")
n_steps = len(test_data)  # 使用实际测试集的长度
print(f"预测步数: {n_steps}")
simulations = np.zeros((100, n_steps))

for i in range(100):
    sim = model_fit.simulate(n_steps, anchor='end')
    simulations[i] = sim

# 计算预测值和置信区间
predicted_mean = np.mean(simulations, axis=0)
confidence_intervals = np.percentile(simulations, [2.5, 97.5], axis=0).T

print(f"预测值长度: {len(predicted_mean)}")

# 如果使用了对数差分转换，需要将预测结果转换回原始尺度
if not is_stationary:
    # 将差分预测值累加回对数尺度
    log_predictions = np.zeros_like(predicted_mean)
    log_predictions[0] = df['Log_Close'][train_size-1] + predicted_mean[0]
    for t in range(1, len(predicted_mean)):
        log_predictions[t] = log_predictions[t-1] + predicted_mean[t]
    
    # 转换回原始尺度
    predicted_mean = np.exp(log_predictions)
    
    # 转换置信区间
    lower_ci = np.zeros_like(confidence_intervals[:, 0])
    upper_ci = np.zeros_like(confidence_intervals[:, 1])
    
    lower_ci[0] = np.exp(df['Log_Close'][train_size-1] + confidence_intervals[0, 0])
    upper_ci[0] = np.exp(df['Log_Close'][train_size-1] + confidence_intervals[0, 1])
    
    for t in range(1, len(confidence_intervals)):
        lower_ci[t] = np.exp(np.log(lower_ci[t-1]) + confidence_intervals[t, 0])
        upper_ci[t] = np.exp(np.log(upper_ci[t-1]) + confidence_intervals[t, 1])
    
    confidence_intervals = np.column_stack([lower_ci, upper_ci])
    
    # 使用原始收盘价作为测试数据
    test_data = df['Close'][original_test_index]

# 创建预测结果的时间索引
forecast_index = original_test_index
print(f"预测索引长度: {len(forecast_index)}")

# 将预测结果转换为Series
predicted_mean = pd.Series(predicted_mean, index=forecast_index)
confidence_intervals = pd.DataFrame(confidence_intervals, 
                                  columns=['lower', 'upper'], 
                                  index=forecast_index)

# 计算预测误差
mse = mean_squared_error(test_data, predicted_mean)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data, predicted_mean)
r2 = r2_score(test_data, predicted_mean)

print(f"\n预测评估指标:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"R² 分数: {r2:.4f}")

# 绘制预测结果
plt.figure(figsize=(15, 8))

# 绘制训练数据、测试数据和预测结果
if not is_stationary:
    plt.plot(df['Close'][:train_size].index, df['Close'][:train_size], label='训练数据', color='blue')
else:
    plt.plot(train_data.index, train_data, label='训练数据', color='blue')
plt.plot(test_data.index, test_data, label='实际测试数据', color='orange')
plt.plot(forecast_index, predicted_mean, label='预测数据', color='red')

# 添加置信区间
plt.fill_between(forecast_index,
                 confidence_intervals['lower'],
                 confidence_intervals['upper'],
                 color='red',
                 alpha=0.1,
                 label='95%置信区间')

plt.title('谷歌股价SARIMAX模型预测', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('股价', fontsize=12)
plt.legend(prop={'size': 10}, loc='upper left')
plt.grid(True)

# 保存图表
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'finance_reports/prediction_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()

# 保存预测结果到CSV
predictions_df = pd.DataFrame({
    'Date': forecast_index,
    'Actual': test_data,
    'Predicted': predicted_mean,
    'Lower_CI': confidence_intervals['lower'],
    'Upper_CI': confidence_intervals['upper']
})
predictions_df.to_csv(f'finance_reports/predictions_{timestamp}.csv', index=True)

# 保存预测报告
with open(f'finance_reports/forecast_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
    f.write(f"谷歌股价SARIMAX模型预测报告\n")
    f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"数据处理信息:\n")
    f.write(f"- 原始数据平稳性: {'是' if is_stationary else '否'}\n")
    if not is_stationary:
        f.write(f"- 使用对数转换和差分处理\n")
    f.write(f"\n模型信息:\n")
    f.write(f"- 模型类型: SARIMAX\n")
    f.write(f"- 非季节性参数 (p,d,q): {order}\n")
    f.write(f"- 季节性参数 (P,D,Q,s): {seasonal_order}\n")
    f.write(f"- 使用Monte Carlo模拟生成预测\n\n")
    f.write("技术指标:\n")
    for feature in features:
        f.write(f"- {feature}\n")
    f.write("\n预测评估指标:\n")
    f.write(f"均方误差 (MSE): {mse:.4f}\n")
    f.write(f"均方根误差 (RMSE): {rmse:.4f}\n")
    f.write(f"平均绝对误差 (MAE): {mae:.4f}\n")
    f.write(f"R² 分数: {r2:.4f}\n\n")
    f.write(f"预测结果已保存到: predictions_{timestamp}.csv\n")
    f.write(f"预测图表已保存到: prediction_plot_{timestamp}.png\n")
    f.write("\n注：预测结果包含95%置信区间，使用Monte Carlo模拟生成\n")

print(f"\n预测报告已保存到: finance_reports/forecast_report_{timestamp}.txt")
print(f"预测结果已保存到: finance_reports/predictions_{timestamp}.csv")
print(f"预测图表已保存到: finance_reports/prediction_plot_{timestamp}.png")
