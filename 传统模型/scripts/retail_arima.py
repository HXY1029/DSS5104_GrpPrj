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
plt.style.use('default')  # 使用默认样式
sns.set_style("whitegrid")  # 设置seaborn样式
mpl.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

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
        ax.plot(actual.index, actual.values, label='实际销售量', linewidth=2, color='#2878B5')
        ax.plot(predicted.index, predicted.values, label='预测销售量', linewidth=2, color='#C82423', linestyle='--')
        
        # 设置标题和标签（使用指定字体）
        if font is not None:
            ax.set_title('零售销售量预测分析', fontproperties=font, fontsize=16, pad=20)
            ax.set_xlabel('日期', fontproperties=font, fontsize=12, labelpad=10)
            ax.set_ylabel('销售量', fontproperties=font, fontsize=12, labelpad=10)
            legend = ax.legend(loc='best', prop=font, fontsize=12, frameon=True, framealpha=0.8)
        else:
            ax.set_title('Retail Sales Prediction', fontsize=16, pad=20)
            ax.set_xlabel('Date', fontsize=12, labelpad=10)
            ax.set_ylabel('Sales', fontsize=12, labelpad=10)
            legend = ax.legend(loc='best', fontsize=12, frameon=True, framealpha=0.8)
        
        # 调整x轴日期显示
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
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
    """
    网格搜索最优SARIMAX参数
    """
    print("正在搜索最优模型参数...")
    
    # 定义参数范围
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]  # 使用7天作为季节性周期
    
    best_aic = float('inf')
    best_params = None
    best_seasonal_params = None
    
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            try:
                model = SARIMAX(train_data,
                              exog=exog_data,
                              order=param,
                              seasonal_order=seasonal_param,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                results = model.fit()
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = param
                    best_seasonal_params = seasonal_param
                    
            except Exception as e:
                continue
    
    print('最优SARIMAX参数: {}x{}'.format(best_params, best_seasonal_params))
    return best_params, best_seasonal_params

def train_sarimax(data, exog_data, order=(1,1,1), seasonal_order=(1,1,1,7)):
    """训练SARIMAX模型"""
    model = SARIMAX(data,
                    exog=exog_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit()
    return model_fit

def prepare_features(df):
    """准备模型特征"""
    # 创建特征
    df['lag_1'] = df['Sales'].shift(1)
    df['lag_7'] = df['Sales'].shift(7)
    df['rolling_mean_7'] = df['Sales'].rolling(window=7).mean()
    df['rolling_std_7'] = df['Sales'].rolling(window=7).std()
    
    # 处理缺失值
    df['lag_1'] = df['lag_1'].fillna(method='bfill')
    df['lag_7'] = df['lag_7'].fillna(method='bfill')
    df['rolling_mean_7'] = df['rolling_mean_7'].fillna(method='bfill')
    df['rolling_std_7'] = df['rolling_std_7'].fillna(method='bfill')
    
    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
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
            f.write("=== 零售销售量预测分析报告 ===\n")
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

def retail_forecast(data_path):
    """零售数据预测"""
    try:
        print("正在读取数据...")
        df = pd.read_csv(data_path)
        
        # 确保Date列是datetime类型
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 首先按时间分组并聚合，避免重复时间戳
        print("\n正在处理数据...")
        df = df.groupby('Date')['Sales'].sum().reset_index()
        
        # 创建连续的时间索引
        idx = pd.date_range(start=df['Date'].min(),
                          end=df['Date'].max(),
                          freq='D')
        
        # 重新索引并填充缺失值
        df = df.set_index('Date')
        df = df.reindex(idx)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 准备特征
        print("\n正在准备特征...")
        exog_features = prepare_features(df)
        
        # 划分训练集和测试集
        print("\n正在划分训练集和测试集...")
        train_size = int(len(df) * 0.8)
        train_data = df['Sales'][:train_size]
        test_data = df['Sales'][train_size:]
        train_exog = exog_features[:train_size]
        test_exog = exog_features[train_size:]
        
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
        feature_names = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
        coefficients = model.params[1:5]  # 跳过截距项
        for name, coef in zip(feature_names, coefficients):
            print(f"{name}: {coef:.4f}")
        
        # 创建输出目录
        output_dir = Path('retail_reports')
        
        # 保存预测结果和报告
        timestamp = save_report(
            predictions=predicted_mean,
            metrics=metrics,
            model_summary=model.summary(),
            feature_importance=dict(zip(
                ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7'],
                model.params[1:5]
            )),
            output_dir=output_dir
        )
        
        # 绘制并保存预测结果图表
        if timestamp:
            plot_path = output_dir / f'prediction_plot_{timestamp}.png'
            plot_predictions(test_data, predicted_mean, '零售销售量预测结果（带特征）', save_path=plot_path)
        
        return predicted_mean, metrics
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        return None, None

if __name__ == "__main__":
    try:
        print("\n=== 零售预测 ===")
        retail_predictions, retail_metrics = retail_forecast('rossmann_normalized.csv')
        if retail_metrics:
            print("\n零售预测评估结果:")
            for metric, value in retail_metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            print("零售预测失败，请检查数据格式")
    except KeyboardInterrupt:
        print('\n程序被用户中断')
        sys.exit(0)
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        sys.exit(1)
