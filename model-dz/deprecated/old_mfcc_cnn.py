import torchaudio
import matplotlib.pyplot as plt
import IPython.display as ipd
import os
import numpy as np
import torch
import torch.nn as nn
import datetime
from pathlib import Path

SAMPLE_RATE = 16000
SEED = 23333

if torch.cuda.is_available():
    GPU_ID = 0
    DEVICE = torch.device(f"cuda:{GPU_ID}")
else:
    DEVICE = torch.device("cpu")


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


def create_mfccs(data, mfcc_transformer):
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


def get_dataloaders(x, y, train_size=0.7, val_size=0.1, batch_size=32):
    num_samples = x.shape[0]
    split1 = int(num_samples * train_size)
    split2 = int(num_samples * (train_size + val_size))

    # shuffle
    indices = np.random.permutation(np.arange(num_samples))
    x = x[indices]
    y = y[indices]

    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)

    x_train = x[:split1]
    x_val = x[split1:split2]
    x_test = x[split2:]

    y_train = y[:split1]
    y_val = y[split1:split2]
    y_test = y[split2:]

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


def onehot_decode(label):
    return torch.argmax(label, dim=1)


def accuracy(predictions, targets):
    pred_decode = onehot_decode(predictions)
    true_decode = targets

    assert len(pred_decode) == len(true_decode)

    acc = torch.mean((pred_decode == true_decode).float())

    return float(acc)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def combine_conv_bn(self):
        conv_result = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )

        scales = self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)
        conv_result.bias[:] = (
            self.conv.bias - self.bn.running_mean
        ) * scales + self.bn.bias
        for ch in range(self.out_channels):
            conv_result.weight[ch, :, :, :] = self.conv.weight[ch, :, :, :] * scales[ch]

        return conv_result


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    batch_acc_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model.forward(x_batch)
        loss = criterion.forward(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        acc = accuracy(out_batch, y_batch)
        batch_acc_list.append(acc)

    return np.mean(batch_loss_list), np.mean(batch_acc_list)


def train_one_epoch(model, trainset_loader, optimizer, criterion):
    model.train()
    batch_loss_list = []
    batch_acc_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model.forward(x_batch)
        loss = criterion.forward(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        acc = accuracy(out_batch, y_batch)
        batch_acc_list.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(batch_loss_list), np.mean(batch_acc_list)


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    criterion,
    max_epochs=100,
    early_stop=10,
    verbose=1,
    plot=False,
    log="train.log",
):
    if log:
        log = open(log, "a")
        log.seek(0)
        log.truncate()

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(
            model, trainset_loader, optimizer, criterion
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        val_loss, val_acc = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if (epoch + 1) % verbose == 0:
            print(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                "\tTrain Loss = %.5f" % train_loss,
                "Train acc = %.5f " % train_acc,
                "Val Loss = %.5f" % val_loss,
                "Val acc = %.5f " % val_acc,
            )

            if log:
                print(
                    datetime.datetime.now(),
                    "Epoch",
                    epoch + 1,
                    "\tTrain Loss = %.5f" % train_loss,
                    "Train acc = %.5f " % train_acc,
                    "Val Loss = %.5f" % val_loss,
                    "Val acc = %.5f " % val_acc,
                    file=log,
                )
                log.flush()

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()
        else:
            wait += 1
            if wait >= early_stop:
                print(f"Early stopping at epoch: {epoch+1}")
                print(f"Best at epoch {best_epoch+1}:")
                print(
                    "Train Loss = %.5f" % train_loss_list[best_epoch],
                    "Train acc = %.5f " % train_acc_list[best_epoch],
                )
                print(
                    "Val Loss = %.5f" % val_loss_list[best_epoch],
                    "Val acc = %.5f " % val_acc_list[best_epoch],
                )

                if log:
                    print(f"Early stopping at epoch: {epoch+1}", file=log)
                    print(f"Best at epoch {best_epoch+1}:", file=log)
                    print(
                        "Train Loss = %.5f" % train_loss_list[best_epoch],
                        "Train acc = %.5f " % train_acc_list[best_epoch],
                        file=log,
                    )
                    print(
                        "Val Loss = %.5f" % val_loss_list[best_epoch],
                        "Val acc = %.5f " % val_acc_list[best_epoch],
                        file=log,
                    )
                    log.flush()
                break

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.plot(range(0, epoch + 1), train_acc_list, "-", label="Train Acc")
        plt.plot(range(0, epoch + 1), val_acc_list, "-", label="Val Acc")
        plt.title("Epoch-Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    if log:
        log.close()

    # torch.save(best_state_dict, "./saved/best_state_dict.pkl")
    model.load_state_dict(best_state_dict)
    return model


class SimpleCLS(nn.Module):
    def __init__(self, input_size=64, num_cls=2):
        super(SimpleCLS, self).__init__()

        self.input_size = input_size

        self.backbone = nn.Sequential(
            ConvBNReLU(1, 8, 3, 2, 1),  # 64 -> 32
            nn.MaxPool2d(2, 2),  # 32 -> 16
            ConvBNReLU(8, 16, 3, 1),  # 16 -> 14
            nn.MaxPool2d(2, 2),  # 14 -> 7
            ConvBNReLU(16, 16, 3, 2, 1),  # 7 -> 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 4 * 4, out_features=num_cls, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.set_params()
        self.train_phase()

    def set_params(self):
        for m in self.backbone.children():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def train_phase(self):
        self.phase = "train"

    def test_phase(self):
        self.phase = "test"

    def forward(self, x):
        out = self.backbone(x)
        # out = self.classifier(out.view(x.size(0), -1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return self.softmax(out) if self.phase == "test" else out


if __name__ == "__main__":
    num_speakers = 100
    data_all = []
    for i in range(1, 1 + num_speakers):
        data_all.append(
            load_audio_files(f"../LibriSpeech-SI/train/spk{str(i).zfill(3)}/")
        )

    # data_spk1 = load_audio_files("../LibriSpeech-SI/train/spk001/")
    # data_spk2 = load_audio_files("../LibriSpeech-SI/train/spk002/")
    # data_spk3 = load_audio_files("../LibriSpeech-SI/train/spk003/")

    audio_time = 3  # s
    offset_time = audio_time / 2  # s

    window_size = int(SAMPLE_RATE * audio_time)  # 1s 16000个采样点 所以一个window 3s 48000个点
    offset = int(SAMPLE_RATE * offset_time)

    mfcc_transformer = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=64,
        melkwargs={
            "n_fft": 750,
            "hop_length": 750,
            "n_mels": 64,
            "center": False,
            "normalized": True,
        },
    )

    xs = []
    ys = []
    for i, data in enumerate(data_all):
        mfccs = create_mfccs(data, mfcc_transformer)
        xs += mfccs
        ys += [i] * len(mfccs)

    xs = np.array(xs)
    ys = np.array(ys)

    batch_size = 32
    max_epochs = 100
    lr = 0.001
    log_file = "temp.log"

    train_loader, val_loader, test_loader = get_dataloaders(
        xs, ys, batch_size=batch_size, train_size=0.7, val_size=0.1
    )

    model = SimpleCLS(num_cls=num_speakers).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        max_epochs=max_epochs,
        early_stop=10,
        verbose=1,
        plot=False,
        log=log_file,
    )

    # model.test_phase()
    test_loss, test_acc = eval_model(model, test_loader, criterion)
    print("Test Loss = %.5f" % test_loss, "Test acc = %.5f " % test_acc)
    with open(log_file, "a") as f:
        print("Test Loss = %.5f" % test_loss, "Test acc = %.5f " % test_acc, file=f)
