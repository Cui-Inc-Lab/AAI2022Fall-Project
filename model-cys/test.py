import torchvision
import numpy as np
import os

# a = torchvision.models.resnet34()


file_path = "/data/cui/data/LibriSpeech-SI/mfcc_250.npz"
if os.path.exists(file_path):
    print("Loading cached data...")
    npz = np.load(file_path, allow_pickle=True)
    x = npz["x"].item()
    y = npz["y"].item()

print('eee')