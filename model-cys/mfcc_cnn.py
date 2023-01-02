import numpy as np
import torch
import torch.nn as nn


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


class MFCCNN(nn.Module):
    def __init__(self, input_size=64, num_cls=2):
        super(MFCCNN, self).__init__()

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
