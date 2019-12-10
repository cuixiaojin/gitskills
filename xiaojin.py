import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms,models
from cui import deep_learning
from cui import accuracy_test
from cui import network_loading
from cui import network_saving
import torch.nn.functional as F
import os
from collections import OrderedDict



def main():

    # 数据库输入
    train_dir = './network/datas/train'
    test_dir = './network/datas/test'

    # TODO: 定义培训集、验证集和测试集的转换
    train_transforms = transforms.Compose([transforms.Resize([224,224]),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5),
                                                                (0.5, 0.5, 0.5))])

    test_valid_transforms = transforms.Compose([transforms.Resize([224,224]),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5),
                                                                     (0.5, 0.5, 0.5))])

    # TODO: 使用ImageFolder加载数据集
    train_data = datasets.ImageFolder(root = os.path.join(train_dir), transform=train_transforms)
    test_data = datasets.ImageFolder(root = os.path.join(test_dir), transform=test_valid_transforms)

    # TODO: 使用图像数据集和训练表，定义数据加载器
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1)



    fmodel = models.vgg.vgg16(pretrained = True)
    for param in fmodel.parameters():
        param.require_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(4096, 1000)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(1000, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    fmodel.classifier = classifier


    network_loading(fmodel, './network/1.pth')
    # 定制标准和优化器

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(fmodel.parameters(), lr=0.001, momentum=0.9)

    deep_learning(fmodel, trainloader, 1, criterion, optimizer)
    accuracy_test(fmodel,testloader)



    network_saving(fmodel)


if __name__ == "__main__":
    main()