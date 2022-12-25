import torch
import torch.nn as nn
import torchaudio
import os
import numpy as np
from pathlib import Path


def onehot_decode(label):
    return torch.argmax(label, dim=1)


def accuracy(predictions, targets):
    pred_decode = onehot_decode(predictions)
    true_decode = targets

    assert len(pred_decode) == len(true_decode)

    acc = torch.mean((pred_decode == true_decode).float())

    return float(acc)


def load_audio_files(path):
    dataset = []
    walker = sorted(str(p) for p in Path(path).glob(f"*.flac"))

    for i, file_path in enumerate(walker):
        path, filename = os.path.split(file_path)
        waveform, sample_rate = torchaudio.load(file_path)
        dataset.append(waveform)

    return dataset


def slide_window(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    signal_length = signal.shape[axis]

    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(
            frame_length - frames_overlap
        )
        pad_size = int(frame_length - rest_samples)
        if pad_size != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = nn.functional.pad(signal, pad_axis, "constant", pad_value)

    frames = signal.unfold(axis, frame_length, frame_step)
    return frames


def minmax_scale(x):
    return (x - x.min()) / (x.max() - x.min())


def data_split(data, train_size=0.7, val_size=0.1):
    num_samples = len(data)
    split1 = int(num_samples * train_size)
    split2 = int(num_samples * (train_size + val_size))

    np.random.shuffle(data)

    x_train = data[:split1]
    x_val = data[split1:split2]
    x_test = data[split2:]

    return x_train, x_val, x_test


def create_mfccs(data, mfcc_transformer, window_size, offset):
    mfcc_list = []
    for waveform in data:
        pad = False
        if waveform.shape[-1] < window_size:
            pad = True
        sub_waveforms = slide_window(
            waveform, window_size, offset, pad_end=pad
        ).squeeze(0)

        for sub_waveform in sub_waveforms:
            mfcc = mfcc_transformer(sub_waveform[None, :])  # (1, 64, 64)
            mfcc_list.append(mfcc.numpy())

    return mfcc_list


def shuffle_xy_totensor(x, y):
    assert len(x) == len(y)

    indices = np.random.permutation(np.arange(len(x)))
    return torch.FloatTensor(x[indices]), torch.LongTensor(y[indices])


def get_dataloaders(x: dict, y: dict, batch_size=64):
    x_train, x_val, x_test = x.values()
    y_train, y_val, y_test = y.values()

    x_train, y_train = shuffle_xy_totensor(x_train, y_train)
    x_val, y_val = shuffle_xy_totensor(x_val, y_val)
    x_test, y_test = shuffle_xy_totensor(x_test, y_test)

    print(f"Trainset:\tx-{x_train.size()}\ty-{y_train.size()}")
    print(f"Valset:  \tx-{x_val.size()}\ty-{y_val.size()}")
    print(f"Testset:\tx-{x_test.size()}\ty-{y_test.size()}")

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    valset = torch.utils.data.TensorDataset(x_val, y_val)
    testset = torch.utils.data.TensorDataset(x_test, y_test)

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=True
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader
