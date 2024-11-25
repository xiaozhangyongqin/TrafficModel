"""
 @Author: zhangyq
 @FileName: data_prepare.py
 @DateTime: 2024/10/12 13:47
 @SoftWare: PyCharm
"""
import pickle
import os
import numpy as np
import pandas as pd
from .utils import vrange, StandardScaler, print_log
import torch




def load_pkl(pkl_path):
    try:
        with open(pkl_path, "rb") as f:
            plk_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Load data from {pkl_path} falied: Error:{e}")
        raise
    return plk_data


def get_dataloader_from_index_data(
        data_dir,
        tod=True,
        dow=True,
        dom=False,
        batch_size=64,
        log=None,
        history_seq_length=12,
        pred_seq_length=12,
        train_ratio=0.6,
        valid_ratio=0.2,
        shift=False,

):
    data_name = "data_shift" if shift else "data"
    if os.path.isfile(os.path.join(data_dir, f"{data_name}.npz")) == True:
        data = np.load(os.path.join(data_dir, f"{data_name}.npz"))['data'].astype(
            np.float32
        )
    else:  # if datatype is not npz, then load h5 file,need to handle the data
        df = (
            pd.read_hdf(os.path.join(data_dir, f"{data_name}.h5")).fillna(0).astype(int)
        )
        print(df.index.values.dtype)
        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)

        feature_list = [data]
        if tod:
            tod_ind = (
                df.index.values - df.index.values.astype("datetime64[D]")
            ) / np.timedelta64(1, "D")  # time of day
            # np.tile(A, reps), 将A沿着指定的方向复制reps次，reps可以是一个列表
            time_of_day = np.tile(tod_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            feature_list.append(time_of_day)
        if dow:
            dow_tiled = np.tile(df.index.dayofweek, [1, num_nodes, 1]).transpose((2, 1, 0))
            day_of_week = dow_tiled
            feature_list.append(day_of_week)
        if dom:
            dom_tiled = np.tile(df.index.day, [1, num_nodes, 1]).transpose((2, 1, 0))
            day_of_month = dom_tiled
            feature_list.append(day_of_month)
        data = np.concatenate(feature_list, axis=-1)
        # save the data to npz file
        np.savez_compressed(os.path.join(data_dir, f"{data_name}.npz"), data=data)
    # data shape: (num_samples, num_nodes, num_features)
    l, n, f = data.shape
    num_samples = l - history_seq_length - pred_seq_length + 1
    train_samples = round(num_samples * train_ratio)
    valid_samples = round(num_samples * valid_ratio)
    test_samples = num_samples - train_samples - valid_samples

    # 索引列表
    index_list = np.array(
        [
            (t - history_seq_length, t, t + pred_seq_length)
            for t in range(history_seq_length, num_samples + history_seq_length)
        ]
    )
    train_index = index_list[:train_samples]
    val_index = index_list[train_samples:train_samples + valid_samples]
    test_index = index_list[
        train_samples + valid_samples:train_samples + valid_samples + test_samples
    ]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]  # 只预测第一个特征
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"TrainSet: \tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"ValidSet: \tx-{x_val.shape}\ty-{y_val.shape}", log=log)
    print_log(f"TestSet: \tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    # try:
    #     _, _, adj_mx = load_pkl(os.path.join(data_dir, "adj_mx.pkl"))
    # except ValueError:
    #     load_pkl(os.path.join(data_dir, "adj_mx.pkl"))
    # 获取归一化的邻接矩阵
    # adj_mx = normalize_adj_mx(adj_mx, "normlap")

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    train_dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    # return train_dataloader, val_dataloader, test_dataloader, adj_mx, scaler
    adj_mx = np.eye(170)
    return train_dataloader, val_dataloader, test_dataloader, scaler, adj_mx


if __name__ == "__main__":
    data_dir = "../data/PEMS08"
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader_from_index_data(data_dir)
    print(train_dataloader)
    print(val_dataloader)
    print(test_dataloader)
    print(scaler)
    data_path = "../data/PEMS08/adj_mx.pkl"
    data = load_pkl(data_path)
    print(data)


