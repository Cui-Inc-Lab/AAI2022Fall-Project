import torch
import torch.nn as nn
import torchaudio
import torchvision
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import math
import random
from mfcc_cnn import MFCCNN
from pathlib import Path

SAMPLE_RATE = 16000
SEED = 23333

AUDIO_DURATION = 3  # s
OFFSET_DURATION = AUDIO_DURATION / 2  # s

# 1s 16000个采样点 所以一个window 3s 48000个点
WINDOW_SIZE = int(SAMPLE_RATE * AUDIO_DURATION)
OFFSET = int(SAMPLE_RATE * OFFSET_DURATION)

MFCC_KWARGS_64 = {
    "n_mfcc": 64,
    "melkwargs": {
        "n_fft": 750,
        "hop_length": 750,
        "n_mels": 64,
        "center": False,
        "normalized": True,
    },
}

MFCC_KWARGS_224 = {
    "n_mfcc": 224,
    "melkwargs": {
        "n_fft": 501,
        "hop_length": 213,
        "n_mels": 224,
        "center": False,
        "normalized": True,
    },
}


def load_noise_files(path):
    dataset = []
    walker = sorted(str(p) for p in Path(path).glob(f"*.wav"))

    for i, file_path in enumerate(walker):
        path, filename = os.path.split(file_path)
        waveform, sample_rate = torchaudio.load(file_path)
        dataset.append(waveform)
        break

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


def load_audio_files(path):
    dataset = []
    walker = sorted(str(p) for p in Path(path).glob(f"*.flac"))

    for i, file_path in enumerate(walker):
        path, filename = os.path.split(file_path)
        waveform, sample_rate = torchaudio.load(file_path)
        dataset.append(waveform)
        break

    return dataset


def data_split(data, train_size=0.7, val_size=0.1):

    num_samples = len(data)
    split1 = int(num_samples * train_size)
    split2 = int(num_samples * (train_size + val_size))

    np.random.shuffle(data)

    x_train = data[:split1]
    x_val = data[split1:split2]
    x_test = data[split2:]

    return x_train, x_val, x_test


print("adding noise to audio")
data_all = []
# data = []
noise_all = load_noise_files(f"/data/cuilab/AAI/LibriSpeech-SI/noise/")
# data.append(load_audio_files(
#     f"/data/cuilab/AAI/LibriSpeech-SI/train/spk001/spk001_002.flac"))
for i in range(1, 1 + 250):
    data_all.append(
        load_audio_files(
            f"/data/cuilab/AAI/LibriSpeech-SI/train/spk001/")
    )
# data_all = data_all[0]


data_addnoise = []
sample_rate = 16000
for speaker in tqdm(data_all):
    spk_list = []
    for spk_aud in speaker:
        num_noise = math.ceil(len(spk_aud[0])/sample_rate/10)
        noise_lst = random.sample(noise_all, 1)
        all_noise = torch.cat(noise_lst, dim=1)
        noise_cut = all_noise[:, :len(spk_aud[0])]
        # 在这里按信噪比增加噪声

        sum_s = torch.sum(spk_aud**2)
        sum_n = torch.sum(noise_cut**2)
        x = np.sqrt(sum_s/(sum_n * pow(10, 15/10)))

        SNR_add = spk_aud + noise_cut*x

        if torch.isnan(SNR_add).all():
            print('SNR_add has nan')
            continue
        # aud_addnoise = spk_aud+noise_cut

        spk_list.append(SNR_add)
    data_addnoise.append(spk_list)


mfcc_transformer = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE, **MFCC_KWARGS_64
)

import numpy as np
from sklearn import preprocessing
from PIL import Image
# import scipy.misc
for i, data in enumerate(tqdm(data_addnoise)):
    x_train_i = data

    mfccs_train = create_mfccs(
        x_train_i, mfcc_transformer, WINDOW_SIZE, OFFSET)
    for data in mfccs_train:
        list = np.array(data).reshape(64,64)
        # list = preprocessing.MinMaxScaler().fit_transform(list)
        
        plt.imshow(list)
        plt.savefig('noise.png')# 图像保存
        plt.show()
        # image = Image.fromarray(list)
        # # image.show()
        # image = image.convert('RGB')
        # image.save("/home/cuilab/AAI2022Fall-Project/model-zyf/www.jpg")
        
        