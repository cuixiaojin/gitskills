import torch
from torch.autograd import Variable


'''
预测训练集的准确率
'''
def accuracy_test(model, dataloader):
    print('开始对测试集预测')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = Variable(images), Variable(labels)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print('the accuracy is :%.2f%%'%(100*correct / total))
    print('测试结束')


# 创建深度学习功能
'''
将训练集数据进行深度学习
'''


def deep_learning(model, trainloader, epochs,  criterion, optimizer):
    print('开始训练')
    epochs = epochs

    steps = 0

    for e in range(epochs):
        running_loss=0
        correct = 0
        total = 0

        for ii, data in enumerate(trainloader):
            images,labels=data

            images, labels = Variable(images), Variable(labels)
            optimizer.zero_grad()

            # forward and backward
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            if ii % 40 == 0 :
                # 测试准确性
                print('EPOCHS : {}/{}'.format(e + 1, epochs),
                      'Loss : {:.4f}'.format(running_loss/40))
                running_loss = 0.0
                print('the accuracy is :%.2f%%'%(100 * correct / total))
    print('训练结束')
    return True



'''
检查加载函数
'''


# def checkpoint loading function
def network_loading(model, ckp_path):
    state_dict = torch.load(ckp_path)
    model.load_state_dict(state_dict,strict=False)
    print('The Network is Loaded')


def network_saving(model):
    torch.save(model.state_dict(), 't.pth')

    print('The Network is Saved')

