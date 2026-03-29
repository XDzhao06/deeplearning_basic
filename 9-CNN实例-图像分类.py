import torch
import torch.nn as nn
from torch.onnx.symbolic_opset12 import dropout
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader    # 数据加载器
import torch.optim as optim                # 优化器
import matplotlib.pyplot as plt
import pandas as pd
import time
from torchsummary import summary

# 准备数据集 CIFAR10包含6w张(32,32,3)的图片，训练集5w张，测试集1w张，10类

BATCH_SIZE = 16
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# todo:准备数据集
def create_dataset():
    # 数据集路径，是否是训练集，数据预处理转成张量数据，是否联网下载
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset

# todo:搭建神经网络
class ImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积池化层
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool1 = nn.MaxPool2d(2, 2, 0)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_features=64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_features=128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool3 = nn.MaxPool2d(2, 2, 0)

        self.dropout = nn.Dropout(p=0.25)

        # 全连接层
        self.linear1 = nn.Linear(128 * 4 * 4, 512)
        self.bnl1 = nn.BatchNorm1d(512,eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.linear2 = nn.Linear(512, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        # 卷积 -> 归一 -> 激活 -> 池化
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))

        # 全连接只能处理二维数据，要将数据拉平(8, 16, 6, 6) -> (8, 576) 一批8张图

        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.bnl1(self.linear1(x)))
        x = self.dropout(x)

        x = torch.relu(self.linear2(x))
        x = self.dropout(x)

        return self.output(x)


def train(train_dataset):

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ImageNet().to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-5, weight_decay=1e-7)

    epochs = 15
    for epoch in range(epochs):
        total_loss, num, correct, start = 0.0, 0, 0, time.time()

        for x, y in train_loader:
            x, y = (t.to(DEVICE) for t in (x, y))

            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            correct += (torch.argmax(y_pred, dim = -1) == y).sum()
            total_loss += loss.item() * len(y)
            num += len(y)

        print(f'epoch:{epoch + 1}, total_loss:{total_loss / num:.5f}, acc:{correct / num}, time:{(time.time() - start):2f}s')
        torch.save(model.state_dict(), f'model/img_model.pth')



def evaluate(test_dataset):
    test_loader = DataLoader(test_dataset, batch_size= BATCH_SIZE,shuffle=False)
    model = ImageNet().to(DEVICE)
    model.eval()
    model.load_state_dict(torch.load('./model/img_model.pth', weights_only = True))
    total_correct, total_num = 0, 0
    with torch.no_grad():
        for x, y in test_loader:

            x, y = (t.to(DEVICE) for t in (x, y))

            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=-1)
            total_correct += (y == y_pred).sum().item()
            total_num += len(y)

    print(f'ACC:{total_correct / total_num:.2f}')


if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    # print(f'训练集：{train_dataset.data.shape}')
    # print(f'测试集：{test_dataset.data.shape}')
    # print(f'数据集类别：{train_dataset.class_to_idx}')
    # summary(model, (3, 32, 32), batch_size=BATCH_SIZE, device='cpu')
    train(train_dataset)
    evaluate(test_dataset)