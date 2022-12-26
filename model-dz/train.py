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

from utils import (
    onehot_decode,
    accuracy,
    load_audio_files,
    slide_window,
    data_split,
    create_mfccs,
    get_dataloaders,
)
from mfcc_cnn import MFCCNN

SAMPLE_RATE = 16000
SEED = 23333

AUDIO_DURATION = 3  # s
OFFSET_DURATION = AUDIO_DURATION / 2  # s

WINDOW_SIZE = int(SAMPLE_RATE * AUDIO_DURATION)  # 1s 16000个采样点 所以一个window 3s 48000个点
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

    model.load_state_dict(best_state_dict)
    return model


def run(model_name: str, num_speakers=None):
    print(datetime.datetime.now())

    if not num_speakers:
        num_speakers = 250  # 1~250
    batch_size = 64
    max_epochs = 100
    lr = 0.001
    log_file = None
    # log_file = "temp.log"

    model_name = model_name.lower()
    if model_name == "mfccnn":
        model = MFCCNN(num_cls=num_speakers).to(DEVICE)

        file_path = f"./data_cache/mfcc_64_{num_speakers}.pkl"
        mfcc_kwargs = MFCC_KWARGS_64

    elif model_name == "resnet":
        model = torchvision.models.resnet18(weights=None)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_speakers)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # change to 1 channel input
        model = model.to(DEVICE)

        file_path = f"./data_cache/mfcc_224_{num_speakers}.pkl"
        mfcc_kwargs = MFCC_KWARGS_224

    else:
        raise NotImplementedError

    if os.path.exists(file_path):
        print("Loading cached data...")
        data_dict = torch.load(file_path)
        x = data_dict["x"]
        y = data_dict["y"]
    else:
        data_all = []
        for i in range(1, 1 + num_speakers):
            data_all.append(
                load_audio_files(f"../LibriSpeech-SI/train/spk{str(i).zfill(3)}/")
            )

        mfcc_transformer = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE, **mfcc_kwargs
        )

        x = {"train": [], "val": [], "test": []}
        y = {"train": [], "val": [], "test": []}
        for i, data in enumerate(data_all):
            x_train_i, x_val_i, x_test_i = data_split(data)

            mfccs_train = create_mfccs(x_train_i, mfcc_transformer, WINDOW_SIZE, OFFSET)
            mfccs_val = create_mfccs(x_val_i, mfcc_transformer, WINDOW_SIZE, OFFSET)
            mfccs_test = create_mfccs(x_test_i, mfcc_transformer, WINDOW_SIZE, OFFSET)

            x["train"] += mfccs_train
            x["val"] += mfccs_val
            x["test"] += mfccs_test

            y["train"] += [i] * len(mfccs_train)
            y["val"] += [i] * len(mfccs_val)
            y["test"] += [i] * len(mfccs_test)

        for k, v in x.items():
            x[k] = np.array(v)
        for k, v in y.items():
            y[k] = np.array(v)

        print("Saving data to cache...")
        # np.savez_compressed(file_path, x=x, y=y)
        torch.save({"x": x, "y": y}, file_path, pickle_protocol=4)

    train_loader, val_loader, test_loader = get_dataloaders(x, y, batch_size=batch_size)

    print("-------------", model._get_name(), "-------------")

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

    test_loss, test_acc = eval_model(model, test_loader, criterion)
    print("Test Loss = %.5f" % test_loss, "Test acc = %.5f " % test_acc)
    if log_file:
        with open(log_file, "a") as f:
            print("Test Loss = %.5f" % test_loss, "Test acc = %.5f " % test_acc, file=f)

    torch.save(
        model.state_dict(),
        f"./saved_model/{model._get_name()}_{num_speakers}_state_dict.pt",
    )


@torch.no_grad()
def predict(model_name, file_list, save_path=None):
    model_name = model_name.lower()
    if model_name == "mfccnn":
        model = MFCCNN(num_cls=250)
        mfcc_kwargs = MFCC_KWARGS_64

    elif model_name == "resnet":
        model = torchvision.models.resnet18(weights=None)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, 250)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # change to 1 channel input

        mfcc_kwargs = MFCC_KWARGS_224

    else:
        raise NotImplementedError

    print(model._get_name(), "Prediction:")

    param_path = f"./saved_model/{model._get_name()}_250_state_dict.pt"
    model.load_state_dict(torch.load(param_path))
    model = model.to(DEVICE)

    mfcc_transformer = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE, **mfcc_kwargs
    )

    pred_labels = []
    for file in tqdm(file_list):
        waveform, _ = torchaudio.load(file)
        pad = False
        if waveform.shape[-1] < WINDOW_SIZE:
            pad = True
        sub_waveforms = slide_window(
            waveform, WINDOW_SIZE, OFFSET, pad_end=pad
        ).squeeze(0)

        mfcc_list = []
        for sub_waveform in sub_waveforms:
            mfcc = mfcc_transformer(sub_waveform[None, :])  # (1, 64, 64)
            mfcc_list.append(mfcc)

        x = torch.stack(mfcc_list).to(DEVICE)
        y_pred = model(x)
        y_pred_decode = onehot_decode(y_pred).cpu().numpy()
        label_pred = np.argmax(np.bincount(y_pred_decode))
        pred_labels.append(label_pred + 1)

    assert len(file_list) == len(pred_labels)
    if save_path:
        with open(save_path, "w") as f:
            for i in range(len(file_list)):
                print(os.path.basename(file_list[i]), pred_labels[i], file=f)

    return np.array(pred_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, required=True)
    parser.add_argument("-n", type=int, required=False, default=250)
    parser.add_argument("-g", type=int, required=True)
    parser.add_argument("-p", action="store_true", required=False)
    parser.add_argument("-f", type=str, required=False)

    args = parser.parse_args()

    GPU_ID = args.g
    if torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{GPU_ID}")
    else:
        DEVICE = torch.device("cpu")

    model_name = args.m
    num_speakers = args.n

    if args.p:
        if args.f:
            print(predict(model_name, [args.f]))
        else:
            test_path = "../LibriSpeech-SI/test/"
            file_list = [
                os.path.join(test_path, file) for file in sorted(os.listdir(test_path))
            ]
            predict(model_name, file_list, save_path="prediction.txt")
    else:
        run(model_name, num_speakers=num_speakers)
