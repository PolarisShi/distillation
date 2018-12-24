# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True,
                                        download=False, transform=transform_train)

testset = torchvision.datasets.CIFAR10(root='./', train=False,
                                       download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F

class _ConvLayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate):
        super(_ConvLayer, self).__init__()
        
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.add_module('relu', nn.ReLU(inplace=True)),
        self.add_module('norm', nn.BatchNorm2d(num_output_features)),
        
        self.drop_rate = drop_rate

    def forward(self, x):
        x = super(_ConvLayer, self).forward(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential()
        self.features.add_module('convlayer1', _ConvLayer(3, 32, 0.0))
        self.features.add_module('maxpool', nn.MaxPool2d(2, 2))
        self.features.add_module('convlayer3', _ConvLayer(32, 64, 0.0))
        self.features.add_module('avgpool', nn.AvgPool2d(2, 2))
        self.features.add_module('convlayer5', _ConvLayer(64, 128, 0.0))
        
        self.classifier = nn.Linear(128, 10)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        features = self.features(x)
        out = F.avg_pool2d(features, kernel_size=8, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


net = CNN()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

import time
for epoch in range(100):
    time_start=time.time()
    running_loss = 0.
    batch_size = 128
    
    for i, data in enumerate(
            torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2), 0):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print('[%d, %5d] loss: %.4f' %(epoch + 1, (i+1)*batch_size, loss.item()))
    
    torch.save(net, 'compare.pkl')
    time_end=time.time()
    print('Time cost:',time_end-time_start, "s")
    
print('Finished Training')

# torch.save(net, 'compare.pkl')
# net = torch.load('compare.pkl')
net.eval()
import time
time_start=time.time()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
time_end=time.time()
print('Time cost:',time_end-time_start, "s")
