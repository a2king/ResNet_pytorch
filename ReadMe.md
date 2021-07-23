# <center>通过深度残差网络进行图像分类（pytorch网络多网络集成配置）</center>

## 简介
本项目通过配置文件修改，实现pytorch的ResNet18, ResNet34, ResNet50, ResNet101, ResNet152网络更替，并通过代码实现自动生成识别所需的标签文件classes.txt（默认使用编码utf-8）。
开发者只需要填写一些基本的元素如数据集地址，图像预处理大小，模型保存地址即可实现模型的训练和调用。


## 配置文件
```
# 配置
net: ResNet50  # 网络模型选择(ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)
class_path: C:/***/classes.txt  # 标签文件路径
width: 32
height: 32
train:
  epoch: 135  # 遍历数据集次数
  pre_epoch: 0  # 定义已经遍历数据集的次数
  batch_size: 256  # 批处理尺寸(batch_size)
  lr: 0.1  # 学习率
  train_data: C:/原图  # 训练集路径
  test_data: C:/原图  # 训练集路径
  is_gpu: False  # 是否使用gpu
  num_workers: 8  # 并行处理数据进程数，根据显存大小自定义，显存越小work数越小
  out_model_path: H:/***/model  #  网络模型保存地址
test:
  model_path: H:/***/net_181.pth  # 测试所用的模型路径
  is_gpu: False  # 是否使用gpu
```

## 训练
```python
# 修改配置文件config.yaml后运行train.py脚本即可
python3 train.py
```

## 测试主要代码解析（可根据自己的需求自行修改）
```python
"""遍历文件夹"""
root = "H:/geetest_word_label/notfind/***"  # 需要测试的图片文件夹地址

with torch.no_grad():
    for dirpath, dirnames, filenames in os.walk(root):
        for filepath in filenames:
            path = os.path.join(dirpath, filepath).replace('\\', '/')
            images = Image.open(path)
            images_t = transform_test(images).unsqueeze(0)
            outputs = net(images_t.to(device))
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            print(classes[predicted.item()])  # 打印图片分类结果
            cv2.imshow("test", cv2.cvtColor(numpy.asarray(images), cv2.COLOR_RGB2BGR))  # 显示当前测试的图片内容
            cv2.waitKey()
```

## 数据集说明
```
data
├── 阿
   └── 1.jpg
   └── 2.jpg
   └── 3.jpg
   └── ...
   └── 99.jpg
├── 你
   └── 1.jpg
   └── 2.jpg
   └── 3.jpg
   └── ...
   └── 99.jpg
├── 我
   └── 1.jpg
   └── 2.jpg
   └── 3.jpg
   └── ...
   └── 99.jpg
├── ...
|
├── 在
   └── 1.jpg
   └── 2.jpg
   └── 3.jpg
   └── ...
   └── 99.jpg
```
本项目提供网络公开的易盾单文子数据集（链接：https://pan.baidu.com/s/1wl45A1ikrd8qQ9cs4AoKjQ  提取码：oqzv）

## 项目声明
本项目提供一种基于残差神经网络进行图像分类的技术实战。项目仅供与个人研究，请勿进行商业操作或攻击网站。