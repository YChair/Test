import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# 护体----------------------------
import random

# 固定随机种子
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

# 确保不使用非确定性算法（影响GPU训练）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 护体----------------------------





# ----------------------------
# 数据预处理
# ----------------------------
data = pd.read_csv("/kaggle/input/metro-m1/Metro_Interstate_Traffic_Volume_modify.csv")

data['date_time'] = pd.to_datetime(data['date_time'])
data['hour'] = data['date_time'].dt.hour
data['day'] = data['date_time'].dt.day
data['month'] = data['date_time'].dt.month
data['day_of_week'] = data['date_time'].dt.dayofweek

features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'day_of_week']
target = ['traffic_volume']

scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

data[features] = scaler_features.fit_transform(data[features])
data[target] = scaler_target.fit_transform(data[target])

X = data[features].values
y = data[target].values

def create_time_series(data, target, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(target[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 12
X, y = create_time_series(X, y, seq_len)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# ----------------------------
# 模型定义
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output

# ----------------------------
# 模型训练与评估
# ----------------------------
input_size = X_train_tensor.shape[2]
hidden_size = 64
num_layers = 2
output_size = 1
epochs = 50
learning_rate = 0.001

model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor.squeeze())
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze()
    test_loss = criterion(predictions, y_test_tensor.squeeze())
    print(f"Test Loss: {test_loss.item()}")

# # ----------------------------
# # 可视化结果
# # ----------------------------
# predictions = scaler_target.inverse_transform(predictions.cpu().numpy().reshape(-1, 1))
# y_test_actual = scaler_target.inverse_transform(y_test_tensor.cpu().numpy())

# plt.figure(figsize=(10, 6))
# plt.plot(y_test_actual[:500], label="Actual")
# plt.plot(predictions[:500], label="Predicted")
# plt.legend()
# plt.title("Traffic Volume Prediction")
# plt.show()



# ----------------------------
# 可视化结果 - 显示 3 天的数据
# ----------------------------

# 反归一化
predictions = scaler_target.inverse_transform(predictions.cpu().numpy().reshape(-1, 1))
y_test_actual = scaler_target.inverse_transform(y_test_tensor.cpu().numpy())

# 提取原始时间戳
timestamps = data['date_time'][-len(y_test_actual):]  # 与测试集对齐
timestamps = timestamps.reset_index(drop=True)

# 提取前 3 天的数据（假设数据为小时级别，每天 24 个数据点）
hours_per_day = 24
days_to_plot = 3
points_to_plot = hours_per_day * days_to_plot
timestamps_to_plot = timestamps[:points_to_plot]
y_test_to_plot = y_test_actual[:points_to_plot]
predictions_to_plot = predictions[:points_to_plot]

# 绘制结果
plt.figure(figsize=(15, 6))
plt.plot(timestamps_to_plot, y_test_to_plot, label="Actual")
plt.plot(timestamps_to_plot, predictions_to_plot, label="Predicted")
plt.xticks(rotation=45)  # 旋转 x 轴标签方便阅读
plt.legend()
plt.title("Traffic Volume Prediction (3 Days)")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.tight_layout()
plt.show()

