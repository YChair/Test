import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path, seq_len=12):
    data = pd.read_csv(file_path)
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].dt.hour
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

    X, y = create_time_series(X, y, seq_len)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler_target
