import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.timefeatures import time_features


class data_detime(Dataset):
    def __init__(self, data, lookback_length, lookforward_length, dataset=None, enc_in=5, timestamp=None,
                 multi_steps=False):
        self.seq_len = lookback_length
        self.pred_len = lookforward_length
        self.multi_steps = multi_steps
        self.data = data[:, :]
        self.enc_in = enc_in
        self.flag = False if timestamp is None else True
        if self.flag:
            self.timestamp = timestamp
            self.timestamp.rename(columns={'timestamp': 'date'}, inplace=True)
            data_stamp = time_features(self.timestamp, timeenc=1, freq='5min' if '5min' in dataset else '15min')
            self.data = np.concatenate((data, data_stamp), axis=1)
        # print(self.data.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        x = self.data[s_begin:s_end] if self.flag else self.data[s_begin:s_end, 0:self.enc_in]
        # x_else = self.data_else[s_begin:s_end]
        if self.multi_steps:
            y = self.data[s_end:s_end + self.pred_len, 0]
        else:
            y = self.data[s_end + self.pred_len - 1:s_end + self.pred_len, 0]
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1


def split_data(data, train, test, lookback_length):
    # data = data.loc[~(data['gen'] == 0)]
    # for column in list(data.columns[data.isnull().sum() > 0]):
    #     data[column].interpolate(method='linear', limit_direction='forward')
    timestamp = data[['timestamp']]
    timestamp['timestamp'] = pd.to_datetime(timestamp.timestamp)
    cols = list(data.columns)
    cols.remove('timestamp')
    data = data[cols].values
    data[:, 0] = np.maximum(data[:, 0], 0)
    length = len(data)
    num_train = train if train > 1 else int(length * train)
    num_test = test if test > 1 else int(length * test)
    num_valid = length - num_test - num_train

    timestamp_train = timestamp[0:num_train]
    timestamp_valid = timestamp[num_train - lookback_length:num_train + num_valid]
    timestamp_test = timestamp[num_train + num_valid - lookback_length:]
    scalar = StandardScaler()
    # scalar = MinMaxScaler()
    scalar_y = StandardScaler()
    # scalar_y = MinMaxScaler()
    y = data[0:num_train, 0].reshape(-1, 1)
    scalar_y.fit(y)
    # y_trans=scalar_y.transform(y)
    # y_re=scalar_y.inverse_transform(y_trans.reshape(-1,1))
    # scalar = MinMaxScaler()
    scalar.fit(data[0:num_train])
    data = scalar.transform(data)
    # data_re=scalar.inverse_transform(data)
    data_train = data[0:num_train]
    data_valid = data[num_train - lookback_length:num_train + num_valid]
    data_test = data[num_train + num_valid - lookback_length:length]

    # plt.plot(range(num_train), data[0:num_train, 0], 'r', range(num_train, num_train + num_valid), data[num_train:num_train + num_valid, 0], 'g',
    #          range(num_train + num_valid, length), data[num_train + num_valid:length, 0], 'b')
    # plt.title(f'{site}-{dataset}')
    # plt.show()
    return data_train, data_valid, data_test, timestamp_train, timestamp_valid, timestamp_test, scalar_y


def split_data_v1(data, train, test, lookback_length):
    # data = data.loc[~(data['gen'] == 0)]
    # for column in list(data.columns[data.isnull().sum() > 0]):
    #     data[column].interpolate(method='linear', limit_direction='forward')
    # timestamp = data[['timestamp']]
    # timestamp['timestamp'] = pd.to_datetime(timestamp.timestamp)
    # cols = list(data.columns)
    # # cols.remove('timestamp')
    # cols_to_convert = data.columns[2:]
    # data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    data = data.iloc[:,2:].values.astype(np.float32)
    data[:, 0] = np.maximum(data[:, 0], 0)
    length = len(data)
    num_train = train if train > 1 else int(length * train)
    num_test = test if test > 1 else int(length * test)
    num_valid = length - num_test - num_train
    #
    # timestamp_train = timestamp[0:num_train]
    # timestamp_valid = timestamp[num_train - lookback_length:num_train + num_valid]
    # timestamp_test = timestamp[num_train + num_valid - lookback_length:]
    scalar = StandardScaler()
    # scalar = MinMaxScaler()
    scalar_y = StandardScaler()
    # scalar_y = MinMaxScaler()
    y = data[0:num_train, 0].reshape(-1, 1)
    scalar_y.fit(y)
    # y_trans=scalar_y.transform(y)
    # y_re=scalar_y.inverse_transform(y_trans.reshape(-1,1))
    # scalar = MinMaxScaler()
    scalar.fit(data[0:num_train])
    data = scalar.transform(data)
    # data_re=scalar.inverse_transform(data)
    data_train = data[0:num_train]
    data_valid = data[num_train - lookback_length:num_train + num_valid]
    data_test = data[num_train + num_valid - lookback_length:length]

    # plt.plot(range(num_train), data[0:num_train, 0], 'r', range(num_train, num_train + num_valid), data[num_train:num_train + num_valid, 0], 'g',
    #          range(num_train + num_valid, length), data[num_train + num_valid:length, 0], 'b')
    # plt.title(f'{site}-{dataset}')
    # plt.show()
    return data_train, data_valid, data_test, None, None, None, scalar_y
