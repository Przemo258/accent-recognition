import torch
import torch.nn as nn
from torchinfo import summary


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ClassificationNet(n_classes=10).to(device)
    summary(model, (10, 1, 128, 188))

    rand = torch.randn(10, 1, 128, 188).to(device)
    output = model(rand)
    print(output.shape)


class Conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, shape=3, pooling=(2, 2), dropout=0.1):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, shape, padding=shape // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


# input (batch_size, channels, mels, samples) : (10, 1, 128, 188)
class ClassificationNet(nn.Module):
    def __init__(self, num_channels=16, flatten_multiplayer=4 * 5, n_classes=10):
        super(ClassificationNet, self).__init__()

        self.input_bn = nn.BatchNorm2d(1)

        self.layer1 = Conv_2d(1, num_channels, pooling=(2, 2))
        self.layer2 = Conv_2d(num_channels, num_channels, pooling=(2, 2))
        self.layer3 = Conv_2d(num_channels, num_channels * 2, pooling=(2, 2))
        self.layer4 = Conv_2d(num_channels * 2, num_channels * 2, pooling=(2, 2))
        self.layer5 = Conv_2d(num_channels * 2, num_channels * 4, pooling=(2, 2))

        self.dense1 = nn.Linear(num_channels * 4 * flatten_multiplayer, num_channels * 4)
        self.dense_bn = nn.BatchNorm1d(num_channels * 4)
        self.dense2 = nn.Linear(num_channels * 4, n_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, out):
        # input batch normalization
        out = self.input_bn(out)

        # convolutional layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        # reshape. (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
        out = out.flatten(1)

        # dense layers
        out = self.dense1(out)
        out = self.dense_bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.dense2(out)

        return out


if __name__ == "__main__":
    main()
