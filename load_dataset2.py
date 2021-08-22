import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os

# 使用GPU
# torch.cuda.set_device(gpu_id)
learning_rate = 0.0001

# 数据集的设置——————————————————————————————————————————————————————————————————————————————————————————————————
root = os.getcwd() + '/data1/'  # 调用图像   os.getcwd() 方法用于返回当前工作目录。


# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')


# 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承torch.utils.data.Dataset
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader()):
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:  # 迭代该列表#按行循环txt文本中内容
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable

        self.imgs = imgs
        self.transform = transform
        self.target_tranform = target_transform
        self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path和label分别获得imgs[index],即刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)  # 按照路径读取照片
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
        return img, label  # return回哪些内容，那在训练循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须写，返回的是数据集的长度，也就是图片的数量，要和loader的长度作区分
        return len(self.imgs)

    # 根据自己定义的那个MyDataset创建数据集


# ---------------------------------------------------数据及读取完毕-----------------------------------------------

# 图像的初始化操作
train_transforms = transforms.compose([
    transforms.RandomResizedCrop((227, 227)),
    transforms.ToTensor(),
])

text_transforms = transforms.Compose([
    transforms.RandomResizedCrop((227, 227)),
    transforms.ToTensor(),
])

# 数据集加载方式设置
train_data = MyDataset(txt=root + 'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(txt=root + 'text.txt', transforms=transforms.ToTensor())
# 调用Dataloader和刚刚创建的数据集，来创建dataloader
train_loader = DataLoader(dataset=train_data, batch_size=6, shuffle=True, number_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=6, shuffle=False, num_workers=4)
print('num_of_trainData:', len(train_data))
print('num_of_testData:', len(test_data))
