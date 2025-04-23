import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import sys
import signal
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import itertools
import warnings
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# 设置中文字体
try:
    font = FontProperties(fname=r'/System/Library/Fonts/PingFang.ttc')  # macOS
except:
    try:
        font = FontProperties(fname=r'C:\Windows\Fonts\SimSun.ttc')  # Windows
    except:
        try:
            font = FontProperties(fname=r'/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')  # Linux
        except:
            print("警告: 未能找到合适的中文字体，图表中的中文可能无法正常显示")
            font = None

# 设置绘图样式
plt.style.use('default')
sns.set_style("whitegrid")
mpl.rcParams['axes.unicode_minus'] = False

def signal_handler(sig, frame):
    """处理键盘中断信号"""
    print('\n程序被用户中断')
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

def safe_plot(func):
    """装饰器：安全地处理绘图操作"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            plt.close()  # 确保图表被关闭
            return result
        except Exception as e:
            print(f"绘图过程中出错: {str(e)}")
            plt.close()  # 确保图表被关闭
            return None
    return wrapper

def evaluate_model(y_true, y_pred):
    """评估模型性能"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print("\n模型评估指标:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

@safe_plot
def plot_predictions(actual, predicted, title, save_path=None):
    """绘制预测结果"""
    try:
        # 创建图表
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 设置背景样式
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 绘制实际值和预测值
        ax.plot(actual.index, actual.values, label='实际需求', linewidth=2, color='#2878B5')
        ax.plot(predicted.index, predicted.values, label='预测需求', linewidth=2, color='#C82423', linestyle='--')
        
        # 设置标题和标签
        if font is not None:
            ax.set_title('能源需求预测分析', fontproperties=font, fontsize=16, pad=20)
            ax.set_xlabel('时间', fontproperties=font, fontsize=12, labelpad=10)
            ax.set_ylabel('需求量', fontproperties=font, fontsize=12, labelpad=10)
            legend = ax.legend(loc='best', prop=font, fontsize=12, frameon=True, framealpha=0.8)
        else:
            ax.set_title('Energy Demand Prediction', fontsize=16, pad=20)
            ax.set_xlabel('Time', fontsize=12, labelpad=10)
            ax.set_ylabel('Demand', fontsize=12, labelpad=10)
            legend = ax.legend(loc='best', fontsize=12, frameon=True, framealpha=0.8)
        
        # 调整x轴日期显示
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        
        # 自动调整x轴标签角度和间距
        plt.xticks(rotation=45, ha='right')
        
        # 设置y轴范围，留出一定边距
        y_min = min(min(actual.values), min(predicted.values))
        y_max = max(max(actual.values), max(predicted.values))
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测结果图表已保存至: {save_path}")
        
        # 显示图表
        plt.show()
        
    except Exception as e:
        print(f"绘图出错: {str(e)}")
        raise
    finally:
        plt.close()

def optimize_sarimax_params(train_data, exog_data):
    """网格搜索最优SARIMAX参数"""
    print("正在搜索最优模型参数...")
    
    # 定义参数范围
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]
    
    best_aic = float('inf')
    best_params = None
    best_seasonal_params = None
    
    # 使用部分数据进行参数搜索
    train_size = min(len(train_data), 5000)
    train_subset = train_data[:train_size]
    exog_subset = exog_data[:train_size] if exog_data is not None else None
    
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            try:
                model = SARIMAX(train_subset,
                              exog=exog_subset,
                              order=param,
                              seasonal_order=seasonal_param,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                results = model.fit(disp=False, maxiter=50)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = param
                    best_seasonal_params = seasonal_param
                    
            except Exception:
                continue
    
    print('最优SARIMAX参数: {}x{}'.format(best_params, best_seasonal_params))
    return best_params, best_seasonal_params

def train_sarimax(data, exog_data, order=(1,1,1), seasonal_order=(1,1,1,24)):
    """训练SARIMAX模型"""
    model = SARIMAX(data,
                    exog=exog_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False, maxiter=100)
    return model_fit

def prepare_features(df):
    """准备模型特征"""
    # 创建时间特征
    df['hour'] = df.index.hour
    df['weekday'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    # 创建滞后特征
    df['lag_1'] = df['nat_demand'].shift(1)
    df['lag_24'] = df['nat_demand'].shift(24)  # 前一天同一时刻
    df['lag_168'] = df['nat_demand'].shift(168)  # 上周同一时刻
    
    # 创建统计特征
    df['rolling_mean_6h'] = df['nat_demand'].rolling(window=6).mean()
    df['rolling_mean_24h'] = df['nat_demand'].rolling(window=24).mean()
    df['rolling_std_24h'] = df['nat_demand'].rolling(window=24).std()
    
    # 处理缺失值
    for col in df.columns:
        if col != 'nat_demand':
            df[col] = df[col].fillna(method='bfill')
            df[col] = df[col].fillna(method='ffill')
    
    # 标准化特征
    scaler = StandardScaler()
    features = ['hour', 'weekday', 'month', 'is_weekend', 
               'lag_1', 'lag_24', 'lag_168',
               'rolling_mean_6h', 'rolling_mean_24h', 'rolling_std_24h']
    df[features] = scaler.fit_transform(df[features])
    
    return df[features].values

def save_report(predictions, metrics, model_summary, feature_importance, output_dir):
    """保存预测报告"""
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存预测结果
        predictions_df = pd.DataFrame({
            '时间': predictions.index,
            '预测值': predictions.values
        })
        predictions_path = os.path.join(output_dir, f'predictions_{timestamp}.csv')
        predictions_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
        
        # 生成报告文本
        report_path = os.path.join(output_dir, f'forecast_report_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 能源需求预测分析报告 ===\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. 模型评估指标\n")
            f.write("-----------------\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            f.write("2. 特征重要性分析\n")
            f.write("-----------------\n")
            for feature, importance in feature_importance.items():
                f.write(f"{feature}: {importance:.4f}\n")
            f.write("\n")
            
            f.write("3. 模型详细信息\n")
            f.write("-----------------\n")
            f.write(str(model_summary))
            f.write("\n\n")
            
            f.write("4. 文件说明\n")
            f.write("-----------------\n")
            f.write(f"- 预测结果: predictions_{timestamp}.csv\n")
            f.write(f"- 预测图表: prediction_plot_{timestamp}.png\n")
            f.write(f"- 详细报告: forecast_report_{timestamp}.txt\n")
        
        print(f"\n报告文件已保存至目录: {output_dir}")
        print(f"- 预测结果: predictions_{timestamp}.csv")
        print(f"- 预测图表: prediction_plot_{timestamp}.png")
        print(f"- 详细报告: forecast_report_{timestamp}.txt")
        
        return timestamp
        
    except Exception as e:
        print(f"保存报告时出错: {str(e)}")
        return None

def energy_forecast(data_path):
    """能源数据预测"""
    try:
        print("正在读取数据...")
        df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
        
        # 预处理数据
        print("\n正在处理数据...")
        df = df.sort_index()
        df = df.asfreq('h')
        
        # 计算测试集大小（三个月）
        test_hours = 24 * 90  # 90天 = 3个月
        
        # 使用最近两年的数据
        df = df.last('730D')  # 取最近两年数据
        print(f"\n数据范围: {df.index.min()} 到 {df.index.max()}")
        print(f"总小时数: {len(df)}")
        
        # 准备特征
        print("\n正在准备特征...")
        exog_features = prepare_features(df)
        
        # 划分训练集和测试集（后三个月作为测试集）
        train_data = df['nat_demand'][:-test_hours]
        test_data = df['nat_demand'][-test_hours:]
        train_exog = exog_features[:-test_hours]
        test_exog = exog_features[-test_hours:]
        
        print("\n数据集划分:")
        print(f"训练集: {len(train_data)} 小时 ({len(train_data)/24:.1f}天)")
        print(f"测试集: {len(test_data)} 小时 ({len(test_data)/24:.1f}天)")
        print(f"训练数据范围: {train_data.index.min()} 到 {train_data.index.max()}")
        print(f"测试数据范围: {test_data.index.min()} 到 {test_data.index.max()}")
        
        # 搜索最优参数
        print("\n正在搜索最优参数...")
        best_params, best_seasonal_params = optimize_sarimax_params(train_data, train_exog)
        
        # 训练模型
        print("\n正在训练模型...")
        model = train_sarimax(train_data, train_exog, 
                            order=best_params, 
                            seasonal_order=best_seasonal_params)
        
        # 在测试集上进行预测
        print("\n正在进行预测...")
        test_predictions = model.get_forecast(steps=len(test_data), exog=test_exog)
        predicted_mean = test_predictions.predicted_mean
        
        # 评估模型
        print("\n正在评估模型性能...")
        metrics = evaluate_model(test_data, predicted_mean)
        
        # 打印模型摘要
        print("\n模型摘要:")
        print(model.summary())
        
        # 分析特征重要性
        print("\n特征重要性:")
        feature_names = ['hour', 'weekday', 'month', 'is_weekend', 
                         'lag_1', 'lag_24', 'lag_168',
                         'rolling_mean_6h', 'rolling_mean_24h', 'rolling_std_24h']
        coefficients = model.params[1:11]  # 跳过截距项
        for name, coef in zip(feature_names, coefficients):
            print(f"{name}: {coef:.4f}")
        
        # 创建输出目录
        output_dir = Path('energy_reports')
        
        # 保存预测结果和报告
        timestamp = save_report(
            predictions=predicted_mean,
            metrics=metrics,
            model_summary=model.summary(),
            feature_importance=dict(zip(
                ['hour', 'weekday', 'month', 'is_weekend', 
                 'lag_1', 'lag_24', 'lag_168',
                 'rolling_mean_6h', 'rolling_mean_24h', 'rolling_std_24h'],
                model.params[1:11]
            )),
            output_dir=output_dir
        )
        
        # 绘制并保存预测结果图表
        if timestamp:
            plot_path = output_dir / f'prediction_plot_{timestamp}.png'
            plot_predictions(test_data, predicted_mean, '能源需求预测结果（带特征）', save_path=plot_path)
        
        return predicted_mean, metrics
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        return None, None

if __name__ == "__main__":
    try:
        print("\n=== 能源需求预测 (SARIMAX模型) ===")
        energy_predictions, energy_metrics = energy_forecast('energy_processed.csv')
        if energy_metrics:
            print("\n能源需求预测评估结果:")
            for metric, value in energy_metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            print("能源需求预测失败，请检查数据格式")
    except KeyboardInterrupt:
        print('\n程序被用户中断')
        sys.exit(0)
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        sys.exit(1)