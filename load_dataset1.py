# 子类化数据
import torch.utils.data
import torch
# 数据处理
from torchvision import transforms


class MyTrainData(torch.utils.data.Dataset)  # 子类化
    def __init__(self, root, tranform=None.train=True):  # 第一步初始化各个变量
    self.root=root
    self.train = train

    def __getitem__(self,idx):  #第二步装载数据，返回[img,label],idx就是一张一张读取
        img = imread(img_path)  #img_path根据自己的数据自定义
        img = torch.from_numpy(img).float()  #需要转成float

        gt = imread(gt_path)  #读取gt，如果是分类问题，可以根据文件夹或命名赋值 0 1
        gt = torch.from_numpy(gt).float()

        return img,gt  #返回 一一对应

    def __len__(self):
        return len(self.imagenumber)  #这个是必须返回的长度