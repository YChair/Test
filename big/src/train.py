import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.model import TrafficTransformer
from src.preprocess import preprocess_data

# 设置随机种子
def set_seed(seed=42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model():
    set_seed()

    # 数据预处理
    file_path = "data/Metro_Interstate_Traffic_Volume_modify.csv"
    X_train, X_test, y_train, y_test, scaler_target = preprocess_data(file_path)

    # 转换为 Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 模型配置
    input_dim = X_train_tensor.shape[2]
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    seq_len = 12
    output_dim = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrafficTransformer(input_dim, embed_dim, num_heads, num_layers, seq_len, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 数据加载器
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 训练
    epochs = 50
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch.squeeze())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")

    print("Training completed!")

if __name__ == "__main__":
    train_model()
