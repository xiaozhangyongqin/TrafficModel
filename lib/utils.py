"""
 @Author: zhangyq
 @FileName: utils.py.py
 @DateTime: 2024/10/12 13:45
 @SoftWare: PyCharm
"""
import numpy as np
import random
import torch
import os
import json



class StandardScaler:
    """
       https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()
        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()


def seed_random(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def set_cpu_num(cpu_num: int):
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def masked_mae_loss(y_pred, y_true, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(y_true)
    else:
        mask = y_true != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, y_pred, y_true, null_val=0.0):
        return masked_mae_loss(y_pred, y_true, null_val)

def masked_huber_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.nn.functional.huber_loss(preds, labels, delta=1, reduction="none")
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedHuberLoss:
    def _get_name(self, delta=0.5):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_huber_loss(preds, labels, null_val)

class CustomJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f"Shape: {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)


def vrange(start, stops):

    """Create ranges of integers for multiple start/stop

        For example:

            >>> starts = [1, 2, 3, 4]
            >>> stops  = [4, 5, 6, 7]
            >>> vrange(starts, stops)
            array([[1, 2, 3],
                   [2, 3, 4],
                   [3, 4, 5],
                   [4, 5, 6]])

        Ref: https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
        """
    stops = np.asarray(stops)
    l = stops - start
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])


if __name__ == '__main__':
    starts = [0, 1, 2]
    stops = [3, 4, 5]
    print(vrange(starts, stops))
