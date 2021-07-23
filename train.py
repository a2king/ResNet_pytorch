#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Created on 2020/11/3 13:38
# Project: 
# @Author: CaoYugang
import os
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import resnet as ResNet
from torch.utils.data import DataLoader, Dataset
import yaml

with open('./config.yaml', 'r', encoding='utf-8') as f_config:
    config_result = f_config.read()
    config = yaml.load(config_result, Loader=yaml.FullLoader)

# 定义是否使用GPU
if config["train"]["is_gpu"]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise Exception("本运行服务器未发现GPU，请确认配置文件")
else:
    device = torch.device("cpu")

# 检查模型保存地址
if not os.path.exists(config["train"]["out_model_path"]):
    raise Exception("检查模型保存地址不存在，请确认配置文件")


class CNNNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img, label = self.imageFolderDataset[index]  # 根据索引index获取该图片
        if self.should_invert:
            img = PIL.ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.Resize((config["width"], config["height"])),
    transforms.RandomHorizontalFlip(0.5 if config["train"]["rotating"] else 0),  # 0.5=>图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),  # 维度转化
])

transform_test = transforms.Compose([
    transforms.Resize((config["width"], config["height"])),
    transforms.ToTensor(),
])

training_dir = config["train"]["train_data"]
train_dataset = torchvision.datasets.ImageFolder(root=training_dir)
test_dir = config["train"]["test_data"]
test_dataset = torchvision.datasets.ImageFolder(root=test_dir)
# 根据标签生成标签集文件
classes = []
with open(config["class_path"], 'w', encoding='utf-8') as f:
    for k in train_dataset.class_to_idx:
        classes.append(k)
        f.write("{}\n".format(k))
classes = tuple(classes)

# 生成训练集
trainset = CNNNetworkDataset(imageFolderDataset=train_dataset, should_invert=False, transform=transform_train)
trainloader = DataLoader(dataset=trainset, batch_size=config["train"]["batch_size"], shuffle=True,
                         num_workers=config["train"]["num_workers"])
# 生成测试集
testset = CNNNetworkDataset(imageFolderDataset=test_dataset, should_invert=False, transform=transform_test)
testloader = DataLoader(dataset=trainset, batch_size=config["train"]["batch_size"], shuffle=True,
                        num_workers=config["train"]["num_workers"])

# 模型定义-ResNet（ResNet18, ResNet34, ResNet50, ResNet101, ResNet152）
if config["net"] == "ResNet18":
    net = ResNet.resnet18(num_classes=classes.__len__()).to(device)
elif config["net"] == "ResNet34":
    net = ResNet.resnet34(num_classes=classes.__len__()).to(device)
elif config["net"] == "ResNet50":
    net = ResNet.resnet50(num_classes=classes.__len__()).to(device)
elif config["net"] == "ResNet101":
    net = ResNet.resnet101(num_classes=classes.__len__()).to(device)
elif config["net"] == "ResNet152":
    net = ResNet.resnet152(num_classes=classes.__len__()).to(device)
else:
    raise Exception("网络模型配置存在问题，请确认配置文件")

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题,此标准将LogSoftMax和NLLLoss集成到一个类中。
optimizer = optim.SGD(net.parameters(), lr=config["train"]["lr"], momentum=0.9,
                      weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

# 训练
if __name__ == "__main__":
    best_acc = 90  # 2 初始化best test accuracy
    print("Start Training, %s !" % config["net"])  # 定义遍历数据集的次数
    for epoch in range(config["train"]["pre_epoch"], config["train"]["epoch"]):  # 从先前次数开始训练
        print('\nEpoch: %d' % (epoch + 1))  # 输出当前次数
        net.train()  # 这两个函数只要适用于Dropout与BatchNormalization的网络，会影响到训练过程中这两者的参数
        # 运用net.train()时，训练时每个min - batch时都会根据情况进行上述两个参数的相应调整，所有BatchNormalization的训练和测试时的操作不同。
        sum_loss = 0.0  # 损失数量
        correct = 0.0  # 准确数量
        total = 0.0  # 总共数量
        for i, data in enumerate(trainloader, 0):  # 训练集合enumerate(sequence, [start=0])用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            # 准备数据  i是序号 data是遍历的数据元素
            length = len(trainloader)  # 训练数量
            inputs, labels = data
            # 假想： inputs是当前输入的图像，label是当前图像的标签，这个data中每一个sample对应一个label
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 清空所有被优化过的Variable的梯度.

            # forward + backward
            outputs = net(inputs)  # 得到训练后的一个输出

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # 进行单次优化 (参数更新).

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # 返回输入张量所有元素的最大值。 将dim维设定为1，其它与输入形状保持一致。
            # 这里采用torch.max。torch.max()的第一个输入是tensor格式，所以用outputs.data而不是outputs作为输入；第二个参数1是代表dim的意思，也就是取每一行的最大值，其实就是我们常见的取概率最大的那个index；第三个参数loss也是torch.autograd.Variable格式。
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d/%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1), length, sum_loss / (i + 1), 100. * correct / total))

        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():  # 没有求导
            correct = 0
            total = 0
            for test_i, data in enumerate(testloader):
                net.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if test_i == 100:
                    break
            print('测试分类准确率为：{}%'.format(round(100 * correct / total, 3)))
            acc = 100. * correct / total
            # 将每次测试结果实时写入acc.txt文件中
            print('Saving model......')
            torch.save(net.state_dict(), '%s/net_%d_%03d.pth' % (config["train"]["out_model_path"], epoch + 1, acc))

            # 记录最佳测试分类准确率并写入best_acc.txt文件中
            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(), '%s/best_net_%03d.pth' % (config["train"]["out_model_path"], best_acc))
    print("Training Finished, TotalEPOCH=%d" % config["train"]["epoch"])
