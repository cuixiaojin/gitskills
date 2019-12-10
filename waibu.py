import torchvision as tv
from network import cui
from network import duiying
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import heapq
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms,models
from collections import OrderedDict
import matplotlib


matplotlib.rcParams['axes.unicode_minus']=False

def name(max5):
    cc = []
    for i in max5:
        n=int(i)
        max_name=duiying.zidian(n)
        cc.append(max_name)
    return cc
transform =tv.transforms.Compose([tv.transforms.Resize([32, 32]), tv.transforms.ToTensor(), tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def predict(img):
    cui.network_loading(net, './network/1.pth')
    img_ = transform(img).unsqueeze(0)

    outputs = net(img_)

    print('图片已加载')
    probability = F.softmax(outputs, dim=1)
    probability1 = probability.tolist()[0]
    max_5 = heapq.nlargest(5, probability1)
    max_5_index = list(map(probability1.index, heapq.nlargest(5, probability1)))

    ming=name(max_5_index)

    zhfont1 = matplotlib.font_manager.FontProperties(fname='./network/ziti/STKAITI.TTF',size=13)
    plt.bar(range(len(max_5)), max_5, color='rgb', tick_label=ming)
    plt.ylabel('概率',fontproperties=zhfont1)
    plt.xlabel("种类",fontproperties=zhfont1)
    for a, b in zip(range(len(max_5)), max_5):
        plt.text(a, b, '%.2f%%' % (b * 100), ha='center', va='bottom', fontsize=10)
    plt.title('最有可能的种类和概率是:'+ming[0],fontproperties=zhfont1, fontsize= 15)

    probability2=[]
    for i in max_5:
        c = '{:.4f}'.format(i)
        probability2.append(c)
    probability3 = {}
    for i in range(5):
        probability3[ming[i]] = probability2[i]
    print(probability3)
    plt.show()
    return probability3

fmodel = models.vgg.vgg16(pretrained = True)
for param in fmodel.parameters():
    param.require_grad = False
classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(4096, 1000)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(1000, 62)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
fmodel.classifier = classifier

net=fmodel













