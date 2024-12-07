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
# 读取数据
data = pd.read_csv('/kaggle/input/metro-m3/Metro_Interstate_Traffic_Volume_modify.csv')

# 转换时间戳为日期时间格式
data['date_time'] = pd.to_datetime(data['date_time'])

# 提取时间特征
data['hour'] = data['date_time'].dt.hour
data['day'] = data['date_time'].dt.day
data['month'] = data['date_time'].dt.month
data['day_of_week'] = data['date_time'].dt.dayofweek

# 选取特征列
features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'day_of_week']
target = ['traffic_volume']

# 数据标准化
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

data[features] = scaler_features.fit_transform(data[features])
data[target] = scaler_target.fit_transform(data[target])

# 转换为 NumPy 数组
X = data[features].values
y = data[target].values

# 创建时间序列
def create_time_series(data, target, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])  # 取连续 seq_len 个时间步
        y.append(target[i+seq_len])  # 预测第 seq_len+1 个时间步的值
    return np.array(X), np.array(y)

seq_len = 12  # 使用过去 12 个时间步预测未来
X, y = create_time_series(X, y, seq_len)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # (batch_size, seq_len, feature_dim)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # (batch_size, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# ----------------------------
# 模型定义
# ----------------------------
class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return torch.relu(self.conv(x))

class SpatialGraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialGraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return torch.relu(self.fc(x))

class STGCN(nn.Module):
    def __init__(self, in_channels, temporal_channels, spatial_channels, output_size):
        super(STGCN, self).__init__()
        self.temporal1 = TemporalConvLayer(in_channels, temporal_channels)
        self.spatial = SpatialGraphConvLayer(temporal_channels, spatial_channels)
        self.temporal2 = TemporalConvLayer(spatial_channels, temporal_channels)
        self.fc = nn.Linear(temporal_channels, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        x = x.permute(0, 2, 1)  # 调整为 (batch_size, feature_dim, seq_len) 以匹配 1D 卷积
        x = self.temporal1(x)
        x = x.permute(0, 2, 1)  # 恢复为 (batch_size, seq_len, feature_dim)
        x = self.spatial(x)
        x = self.temporal2(x.permute(0, 2, 1))  # 再次调整为 (batch_size, feature_dim, seq_len)
        x = torch.mean(x, dim=2)  # 对时间维度求平均
        x = self.fc(x)
        return x

# ----------------------------
# 模型训练与评估
# ----------------------------
# 定义超参数
in_channels = X_train_tensor.shape[2]  # 输入特征数量
temporal_channels = 64
spatial_channels = 32
output_size = 1
epochs = 50
learning_rate = 0.001

# 初始化模型
model = STGCN(in_channels=in_channels,
              temporal_channels=temporal_channels,
              spatial_channels=spatial_channels,
              output_size=output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 模型训练
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)  # (batch_size, 1)
    loss = criterion(outputs.squeeze(), y_train_tensor.squeeze())  # 对齐形状
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 测试集评估
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze()
    test_loss = criterion(predictions, y_test_tensor.squeeze())
    print(f"Test Loss: {test_loss.item()}")

# ----------------------------
# 可视化结果 - 显示 3 天的数据
# ----------------------------

# 反归一化
predictions = scaler_target.inverse_transform(predictions.numpy().reshape(-1, 1))
y_test_actual = scaler_target.inverse_transform(y_test_tensor.numpy())

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
