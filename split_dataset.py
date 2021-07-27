#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Created on 2021/7/27 8:58
import os
import shutil

org_dataset_path = "C:/Users/~/Downloads/over"  # 所有数据集地址
dataset_test_path = "C:/Users/~/Downloads/dataset_res/test"  # 测试数据集保存
dataset_train_path = "C:/Users/~/Downloads/dataset_res/train"  # 训练数据集保存
min_word_num = 2  # 单字最少图片数量（少于两张此字删除）
train_test_ratio = [8, 1]  # 拆分比例

path_dir = {}
for dirpath, dirnames, filenames in os.walk(org_dataset_path):
    for filepath in filenames:
        path = os.path.join(dirpath, filepath).replace('\\', '/')
        word = filepath.split("_")[0]
        if word in path_dir:
            path_dir[word].append(path)
        else:
            path_dir[word] = [path, ]

test_path = {}
train_path = {}
for word in path_dir:
    if len(path_dir[word]) >= min_word_num:
        for index, _path in enumerate(path_dir[word]):
            if index % sum(train_test_ratio) < train_test_ratio[1]:
                if word in test_path:
                    test_path[word].append(_path)
                else:
                    test_path[word] = [_path, ]
            else:
                if word in train_path:
                    train_path[word].append(_path)
                else:
                    train_path[word] = [_path, ]

for word in test_path:
    for _path in test_path[word]:
        if not os.path.exists(os.path.join(dataset_test_path, word)):
            os.mkdir(os.path.join(dataset_test_path, word))
        shutil.copy(_path, os.path.join(dataset_test_path, word, _path.split("/")[-1]))
for word in train_path:
    for _path in train_path[word]:
        if not os.path.exists(os.path.join(dataset_train_path, word)):
            os.mkdir(os.path.join(dataset_train_path, word))
        shutil.copy(_path, os.path.join(dataset_train_path, word, _path.split("/")[-1]))