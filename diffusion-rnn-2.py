import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model.pytorch.dcrnn_model import DCRNNModel

from main import URBAN_MIX_CSV, URBAN_CORE_CSV, ADJ_URBAN_CORE_CSV, ADJ_URBAN_MIX_CSV

batch_size = 64
test_batch_size = 64
num_batches = 64
epoch = 100
epoch_num = 0
lr_decay_ratio = 0.1
test_every_n_epochs=10
epsilon=1e-8
base_lr = 0.01
steps = [20, 30, 40, 50]
max_grad_norm = 5

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

def generate_train_val_test(df, output_dir):
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    print(x.shape[0])
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.23333333333)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    scaler = StandardScaler(x_train[..., 0].mean(), std=x_train[..., 0].std())

    data = {}

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        
        data['x_' + cat] = scaler.transform(_x[..., 0])
        data['y_' + cat] = scaler.transform(_y[..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler

    return data

def load_dataset(file_name):
    df = pd.read_csv(file_name, header=None) 
    df = df.drop(df.columns[0:7], axis=1).reset_index(drop=True)
    
    df = df.transpose()
    print(df.shape)

    data = generate_train_val_test(df, "dataset/")
    return data

def prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

def get_x_y(self, x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                y shape (horizon, batch_size, num_sensor, input_dim)
    """
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    self._logger.debug("X: {}".format(x.size()))
    self._logger.debug("y: {}".format(y.size()))
    x = x.permute(1, 0, 2, 3)
    y = y.permute(1, 0, 2, 3)
    return x, y

def evaluate(self, dataset='val', batches_seen=0):
    """
    Computes mean L1Loss
    :return: mean L1Loss
    """
    with torch.no_grad():
        self.dcrnn_model = self.dcrnn_model.eval()

        val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
        losses = []

        y_truths = []
        y_preds = []

        for _, (x, y) in enumerate(val_iterator):
            x, y = self._prepare_data(x, y)

            output = self.dcrnn_model(x)
            loss = self._compute_loss(y, output)
            losses.append(loss.item())

            y_truths.append(y.cpu())
            y_preds.append(output.cpu())

        mean_loss = np.mean(losses)

        self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

        y_preds = np.concatenate(y_preds, axis=1)
        y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

        y_truths_scaled = []
        y_preds_scaled = []
        for t in range(y_preds.shape[0]):
            y_truth = self.standard_scaler.inverse_transform(y_truths[t])
            y_pred = self.standard_scaler.inverse_transform(y_preds[t])
            y_truths_scaled.append(y_truth)
            y_preds_scaled.append(y_pred)

        return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}


def _compute_loss(self, y_true, y_predicted):
    y_true = self.standard_scaler.inverse_transform(y_true)
    y_predicted = self.standard_scaler.inverse_transform(y_predicted)
    return masked_mae_loss(y_predicted, y_true)

def evaluate(self, dataset='val', batches_seen=0):
    """
    Computes mean L1Loss
    :return: mean L1Loss
    """
    with torch.no_grad():
        self.dcrnn_model = self.dcrnn_model.eval()

        val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
        losses = []

        y_truths = []
        y_preds = []

        for _, (x, y) in enumerate(val_iterator):
            x, y = self._prepare_data(x, y)

            output = self.dcrnn_model(x)
            loss = self._compute_loss(y, output)
            losses.append(loss.item())

            y_truths.append(y.cpu())
            y_preds.append(output.cpu())

        mean_loss = np.mean(losses)

        self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

        y_preds = np.concatenate(y_preds, axis=1)
        y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

        y_truths_scaled = []
        y_preds_scaled = []
        for t in range(y_preds.shape[0]):
            y_truth = self.standard_scaler.inverse_transform(y_truths[t])
            y_pred = self.standard_scaler.inverse_transform(y_preds[t])
            y_truths_scaled.append(y_truth)
            y_preds_scaled.append(y_pred)

        return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}


def train_model(data):
    dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
    # steps is used in learning rate - will see if need to use it?
    min_val_loss = float('inf')
    wait = 0
    optimizer = torch.optim.Adam(dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                        gamma=lr_decay_ratio)

    print('Start training ...')

    # this will fail if model is loaded with a changed batch_size
    num_batches = data['train_loader'].num_batch
    print("num_batches:{}".format(num_batches))

    batches_seen = num_batches * epoch_num

    for epoch_num in range(epoch_num, epochs):
        dcrnn_model = dcrnn_model.train()

        train_iterator = data['train_loader'].get_iterator()
        losses = []

        start_time = time.time()

        for _, (x, y) in enumerate(train_iterator):
            optimizer.zero_grad()

            x, y = prepare_data(x, y)

            output = dcrnn_model(x, y, batches_seen)

            if batches_seen == 0:
                # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                optimizer = torch.optim.Adam(dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

            loss = compute_loss(y, output)

            print(loss.item())

            losses.append(loss.item())

            batches_seen += 1
            loss.backward()

            # gradient clipping - this does it in place
            torch.nn.utils.clip_grad_norm_(dcrnn_model.parameters(), max_grad_norm)

            optimizer.step()
        
        print("epoch complete")
        lr_scheduler.step()
        print("evaluating now!")

        val_loss, _ = evaluate(dataset='val', batches_seen=batches_seen)

        end_time = time.time()

        print('training loss',
                np.mean(losses),
                batches_seen)

        if (epoch_num % log_every) == log_every - 1:
            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                        '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                        np.mean(losses), val_loss, lr_scheduler.get_lr()[0],
                                        (end_time - start_time))
            print(message)

        if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
            test_loss, _ = self.evaluate(dataset='test', batches_seen=batches_seen)
            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                        '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                        np.mean(losses), test_loss, lr_scheduler.get_lr()[0],
                                        (end_time - start_time))
            print(message)

        if val_loss < min_val_loss:
            wait = 0
            if save_model:
                model_file_name = self.save_model(epoch_num)
                print(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(min_val_loss, val_loss, model_file_name))
            min_val_loss = val_loss

        elif val_loss >= min_val_loss:
            wait += 1
            if wait == patience:
                print('Early stopping at epoch: %d' % epoch_num)
                break


if __name__ == "__main__":
    data = load_dataset(URBAN_CORE_CSV)