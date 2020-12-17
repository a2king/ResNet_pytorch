#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Created on 2020/11/3 13:38
# Project: 
# @Author: CaoYugang
import torch
import os
import cv2
import numpy
from torchvision import transforms
from torchvision.models import resnet as ResNet
from PIL import Image
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

transform_test = transforms.Compose([
    transforms.Resize((config["width"], config["height"])),
    transforms.ToTensor(),
])
classes = []
with open(config["class_path"], 'r', encoding='utf-8') as f:
    for k in f.readlines():
        if k.strip():
            classes.append(k.strip())
classes = tuple(classes)

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

if __name__ == "__main__":
    if config["train"]["is_gpu"]:
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(config["test"]["model_path"]))
        else:
            raise Exception("本运行服务器未发现GPU，请确认配置文件")
    else:
        net.load_state_dict(torch.load(config["test"]["model_path"], map_location='cpu'))
    net.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。

    """遍历文件夹"""
    root = "H:/geetest_word_label/notfind/完成"

    with torch.no_grad():  # 没有求导
        for dirpath, dirnames, filenames in os.walk(root):
            for filepath in filenames:
                path = os.path.join(dirpath, filepath).replace('\\', '/')
                images = Image.open(path)
                images_t = transform_test(images).unsqueeze(0)
                outputs = net(images_t.to(device))
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                print(classes[predicted.item()])
                cv2.imshow("test", cv2.cvtColor(numpy.asarray(images), cv2.COLOR_RGB2BGR))
                cv2.waitKey()
