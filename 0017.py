import numpy as np # 导入所需库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler #标准化库

# load data
train_data = pd.read_csv('0017data/0017.HK training.csv')
test_data = pd.read_csv('0017data/0017.HK test.csv')
# # 提取价格列
y_test = test_data['Close'].values
window_size = 5 # 计算移动平均
y_test_5_moving_avg = np.convolve(y_test, np.ones(window_size)/window_size, mode='valid')
y_test = y_test[:-(window_size-1)]

r2_test = round(r2_score(y_test, y_test_5_moving_avg),4) #测试集评价指标计算并保存
mse_test = round(mean_squared_error(y_test, y_test_5_moving_avg),4)
rmse_test = np.sqrt(mse_test)
mae_test = round(mean_absolute_error(y_test, y_test_5_moving_avg),4)
mape_test = round(mean_absolute_percentage_error(y_test, y_test_5_moving_avg),4)
metrics_svr=np.array([r2_test,mse_test,rmse_test,mae_test,mape_test])
pd.DataFrame(metrics_svr,index = ['R2','MSE','RMSE','MAE','MAPE']).to_csv('Metrics_5天移动平均.csv',header=False,index=True)

plt.plot(range(len(y_test)), y_test, label='True values',c='mistyrose') # 绘制预测值和真实值的曲线图
plt.plot(range(len(y_test)), y_test_5_moving_avg, label='Predicted values',c='navy')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.savefig(fname='折线图_5天移动平均.png',dpi=600)
plt.show()
plt.clf()

# 提取价格列
y_test = test_data['Close'].values
window_size = 10 # 计算移动平均
y_test_5_moving_avg = np.convolve(y_test, np.ones(window_size)/window_size, mode='valid')
y_test = y_test[:-(window_size-1)]

r2_test = round(r2_score(y_test, y_test_5_moving_avg),4) #测试集评价指标计算并保存
mse_test = round(mean_squared_error(y_test, y_test_5_moving_avg),4)
rmse_test = np.sqrt(mse_test)
mae_test = round(mean_absolute_error(y_test, y_test_5_moving_avg),4)
mape_test = round(mean_absolute_percentage_error(y_test, y_test_5_moving_avg),4)
metrics_svr=np.array([r2_test,mse_test,rmse_test,mae_test,mape_test])
pd.DataFrame(metrics_svr,index = ['R2','MSE','RMSE','MAE','MAPE']).to_csv('Metrics_10天移动平均.csv',header=False,index=True)

plt.plot(range(len(y_test)), y_test, label='True values',c='mistyrose') # 绘制预测值和真实值的曲线图
plt.plot(range(len(y_test)), y_test_5_moving_avg, label='Predicted values',c='navy')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.savefig(fname='折线图_10天移动平均.png',dpi=600)
plt.show()
plt.clf()

#-------------------5天预测
ty = train_data['Close'].values
x_train = []
y_train = []
for i in range(5, len(ty)):
    x_train.append(ty[i-5:i])
    y_train.append(ty[i])
x_train = np.array(x_train)
y_train = np.array(y_train)

ty = test_data['Close'].values
x_test = []
y_test = []
# 基于前5天的数据构造特征矩阵和目标向量
for i in range(5, len(ty)):
    x_test.append(ty[i-5:i])
    y_test.append(ty[i])
x_test = np.array(x_test)
y_test = np.array(y_test)

model = LinearRegression()
model.fit(x_train, y_train)

y_test_pred = model.predict(x_test) #测试集预测
r2_test = round(r2_score(y_test, y_test_pred),4) #测试集评价指标计算并保存
mse_test = round(mean_squared_error(y_test, y_test_pred),4)
rmse_test = np.sqrt(mse_test)
mae_test = round(mean_absolute_error(y_test, y_test_pred),4)
mape_test = round(mean_absolute_percentage_error(y_test, y_test_pred),4)
metrics_svr=np.array([r2_test,mse_test,rmse_test,mae_test,mape_test])
pd.DataFrame(metrics_svr,index = ['R2','MSE','RMSE','MAE','MAPE']).to_csv('Metrics_5天预测.csv',header=False,index=True)

plt.plot(range(len(y_test)), y_test, label='True values',c='mistyrose') # 绘制预测值和真实值的曲线图
plt.plot(range(len(y_test)), y_test_pred, label='Predicted values',c='navy')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.savefig(fname='折线图_5天预测.png',dpi=600)
plt.show()
plt.clf()

#-------------------10天预测
ty = train_data['Close'].values
x_train = []
y_train = []
for i in range(10, len(ty)):
    x_train.append(ty[i-10:i])
    y_train.append(ty[i])
x_train = np.array(x_train)
y_train = np.array(y_train)

ty = test_data['Close'].values
x_test = []
y_test = []

for i in range(10, len(ty)):
    x_test.append(ty[i-10:i])
    y_test.append(ty[i])
x_test = np.array(x_test)
y_test = np.array(y_test)

model = LinearRegression()
model.fit(x_train, y_train)

y_test_pred = model.predict(x_test) #测试集预测
r2_test = round(r2_score(y_test, y_test_pred),4) #测试集评价指标计算并保存
mse_test = round(mean_squared_error(y_test, y_test_pred),4)
rmse_test = np.sqrt(mse_test)
mae_test = round(mean_absolute_error(y_test, y_test_pred),4)
mape_test = round(mean_absolute_percentage_error(y_test, y_test_pred),4)
metrics_svr=np.array([r2_test,mse_test,rmse_test,mae_test,mape_test])
pd.DataFrame(metrics_svr,index = ['R2','MSE','RMSE','MAE','MAPE']).to_csv('Metrics_10天预测.csv',header=False,index=True)

plt.plot(range(len(y_test)), y_test, label='True values',c='mistyrose') # 绘制预测值和真实值的曲线图
plt.plot(range(len(y_test)), y_test_pred, label='Predicted values',c='navy')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.savefig(fname='折线图_10天预测.png',dpi=600)
plt.show()
plt.clf()

#-------------------基于以前的价格
ty = train_data['Close'].values
x_train = []
y_train = []
for i in range(8, len(ty)):
    x_train.append(ty[i-8:i])
    y_train.append(ty[i])
x_train = np.array(x_train)
y_train = np.array(y_train)

ty = test_data['Close'].values
x_test = []
y_test = []

for i in range(8, len(ty)):
    x_test.append(ty[i-8:i])
    y_test.append(ty[i])
x_test = np.array(x_test)
y_test = np.array(y_test)

model = MLPRegressor(solver='adam', hidden_layer_sizes=(4,),alpha=0.001,
                        batch_size='auto',learning_rate='constant',learning_rate_init=0.001,max_iter=100,random_state=1)
model.fit(x_train, y_train)

y_test_pred = model.predict(x_test) #测试集预测
r2_test = round(r2_score(y_test, y_test_pred),4) #测试集评价指标计算并保存
mse_test = round(mean_squared_error(y_test, y_test_pred),4)
rmse_test = np.sqrt(mse_test)
mae_test = round(mean_absolute_error(y_test, y_test_pred),4)
mape_test = round(mean_absolute_percentage_error(y_test, y_test_pred),4)
metrics_svr=np.array([r2_test,mse_test,rmse_test,mae_test,mape_test])
pd.DataFrame(metrics_svr,index = ['R2','MSE','RMSE','MAE','MAPE']).to_csv('Metrics_MLP_以前.csv',header=False,index=True)

plt.plot(range(len(y_test)), y_test, label='True values',c='mistyrose') # 绘制预测值和真实值的曲线图
plt.plot(range(len(y_test)), y_test_pred, label='Predicted values',c='navy')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.savefig(fname='折线图_MLP_以前.png',dpi=600)
plt.show()
plt.clf()

#基于价格change预测
prices = train_data['Close'].values
# 计算价格变化率
price_diff = np.diff(prices) / prices[:-1]
x_train = []
y_train = []
for i in range(8, len(price_diff)):
    x_train.append(price_diff[i-8:i])
    y_train.append(prices[i])
x_train = np.array(x_train)
y_train = np.array(y_train)


prices = test_data['Close'].values
# 计算价格变化率
price_diff = np.diff(prices) / prices[:-1]
x_test = []
y_test = []
# 基于前5天的数据构造特征矩阵和目标向量
for i in range(8, len(price_diff)):
    x_test.append(price_diff[i-8:i])
    y_test.append(prices[i])
x_test = np.array(x_test)
y_test = np.array(y_test)

x_scale, y_scale = StandardScaler(), StandardScaler() #数据标准化
x_train_scaled = x_scale.fit_transform(x_train) #读取训练集自变量均值和标准差并转化
x_test_scaled = x_scale.transform(x_test) #转化测试集自变量
y_train_scaled = y_scale.fit_transform(y_train.reshape(-1, 1)) #读取训练集因变量均值和标准差并转化
model = MLPRegressor(solver='adam', hidden_layer_sizes=(4,),alpha=0.001,
                        batch_size='auto',learning_rate='constant',learning_rate_init=0.001,max_iter=100,random_state=1)
model.fit(x_train_scaled, y_train_scaled)

y_test_pred = model.predict(x_test_scaled) #测试集预测
y_test_pred = y_scale.inverse_transform(y_test_pred.reshape(-1, 1)) #预测值反标准化
r2_test = round(r2_score(y_test, y_test_pred),4) #测试集评价指标计算并保存
mse_test = round(mean_squared_error(y_test, y_test_pred),4)
rmse_test = np.sqrt(mse_test)
mae_test = round(mean_absolute_error(y_test, y_test_pred),4)
mape_test = round(mean_absolute_percentage_error(y_test, y_test_pred),4)
metrics_svr=np.array([r2_test,mse_test,rmse_test,mae_test,mape_test])
pd.DataFrame(metrics_svr,index = ['R2','MSE','RMSE','MAE','MAPE']).to_csv('Metrics_MLP_变化.csv',header=False,index=True)

plt.plot(range(len(y_test)), y_test, label='True values',c='mistyrose') # 绘制预测值和真实值的曲线图
plt.plot(range(len(y_test)), y_test_pred, label='Predicted values',c='navy')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.savefig(fname='折线图_MLP_变化.png',dpi=600)
plt.show()
plt.clf()