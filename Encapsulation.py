import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms,models
from cui import deep_learning
from cui import network_loading
from cui import network_saving
import os
from collections import OrderedDict

class B():
    def __init__(self):
        self.fmodel = models.vgg.vgg16(pretrained=True)
        for param in self.fmodel.parameters():
            param.require_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(4096, 1000)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(1000, 62)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        self.fmodel.classifier = classifier
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.fmodel.parameters(), lr=0.001, momentum=0.9)
        network_loading(self.fmodel, './network/1.pth')
        self.train_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5),
                                                                    (0.5, 0.5, 0.5))])

    def xunlian(self, file):
        train_dir = file
        train_data = datasets.ImageFolder(root=os.path.join(train_dir), transform=self.train_transforms)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
        deep_learning(self.fmodel, trainloader, 5, self.criterion, self.optimizer)

    def save(self):
        network_saving(self.fmodel)