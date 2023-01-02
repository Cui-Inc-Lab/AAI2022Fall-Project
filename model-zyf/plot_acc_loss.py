import numpy as np
import matplotlib.pyplot as plt

log = open("/home/cuilab/AAI2022Fall-Project/model-zyf/logs/resnet18_250.log", "r")
lines = log.readlines()
for index, line in enumerate(lines):
    lines[index] = line.split(' ')

lines = np.array(lines)

train_loss_list = [float(i) for i in lines[:, 7:8].squeeze().tolist()]
train_acc_list = [float(i) for i in lines[:, 11:12].squeeze().tolist()]
val_loss_list = [float(i) for i in lines[:, 16:17].squeeze().tolist()]
val_acc_list = [float(i) for i in lines[:, 20:21].squeeze().tolist()]

epoch = len(lines)

plt.plot(range(0, epoch), train_loss_list, "-", label="Train Loss")
plt.plot(range(0, epoch), val_loss_list, "-", label="Val Loss")
# plt.yticks(np.arange(min(train_acc_list), max(train_acc_list), step=5))
plt.title("ResNet-Loss-Without-Noise")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig("ResNet_loss.png")
plt.close()

plt.plot(range(0, epoch), train_acc_list, "-", label="Train Acc")
plt.plot(range(0, epoch), val_acc_list, "-", label="Val Acc")
plt.title("ResNet-Accuracy-Without-Noise")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
plt.savefig("ResNet_acc.png")
