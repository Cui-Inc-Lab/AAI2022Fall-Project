import torch
import torch.nn as nn
import torchaudio
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os

from utils import accuracy, load_audio_files, data_split, create_mfccs, get_dataloaders
from mfcc_cnn import MFCCNN

SAMPLE_RATE = 16000
SEED = 23333

if torch.cuda.is_available():
    GPU_ID = 0
    DEVICE = torch.device(f"cuda:{GPU_ID}")
else:
    DEVICE = torch.device("cpu")


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


if __name__ == "__main__":
    num_speakers = 250  # 1~250
    batch_size = 64
    max_epochs = 100
    lr = 0.001
    log_file = "temp.log"

    print(datetime.datetime.now())

    file_path = f"./data_cache/mfcc_{num_speakers}.npz"
    if os.path.exists(file_path):
        print("Loading cached data...")
        npz = np.load(file_path, allow_pickle=True)
        x = npz["x"].item()
        y = npz["y"].item()
    else:
        data_all = []
        for i in range(1, 1 + num_speakers):
            data_all.append(
                load_audio_files(f"../LibriSpeech-SI/train/spk{str(i).zfill(3)}/")
            )

        audio_time = 3  # s
        offset_time = audio_time / 2  # s

        window_size = int(
            SAMPLE_RATE * audio_time
        )  # 1s 16000个采样点 所以一个window 3s 48000个点
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

        x = {"train": [], "val": [], "test": []}
        y = {"train": [], "val": [], "test": []}
        for i, data in enumerate(data_all):
            x_train_i, x_val_i, x_test_i = data_split(data)

            mfccs_train = create_mfccs(x_train_i, mfcc_transformer, window_size, offset)
            mfccs_val = create_mfccs(x_val_i, mfcc_transformer, window_size, offset)
            mfccs_test = create_mfccs(x_test_i, mfcc_transformer, window_size, offset)

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
        np.savez(file_path, x=x, y=y)

    train_loader, val_loader, test_loader = get_dataloaders(x, y, batch_size=batch_size)

    model = MFCCNN(num_cls=num_speakers).to(DEVICE)
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

    # model.test_phase()
    test_loss, test_acc = eval_model(model, test_loader, criterion)
    print("Test Loss = %.5f" % test_loss, "Test acc = %.5f " % test_acc)
    with open(log_file, "a") as f:
        print("Test Loss = %.5f" % test_loss, "Test acc = %.5f " % test_acc, file=f)

    torch.save(model.state_dict(), f"{model._get_name()}_state_dict.pt")
