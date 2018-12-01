# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='./', train=True,
                                        download=False, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./', train=False,
                                       download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=False, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = F.dropout(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netT = torch.load('teacher.pkl')
netT.to(device)
soft_target = torch.tensor([]).to(device)
with torch.no_grad():
    for data in trainloader:
        images, _ = data
        images = images.to(device)
        outputs = netT(images)
        soft_target = torch.cat((soft_target, outputs), 0)
soft_target.to("cpu")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000,
                                          shuffle=False, num_workers=2)

with torch.no_grad():
    for data in trainloader:
        images, labels = data

softset = torch.utils.data.TensorDataset(images, labels, soft_target)


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

criterion2 = nn.KLDivLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
netT.to(device)

import time
for epoch in range(100):
    time_start=time.time()
    running_loss = 0.
    batch_size = 128
    
    alpha = 0.95
    
    for i, data in enumerate(
            torch.utils.data.DataLoader(softset, batch_size=batch_size,
                                        shuffle=True, num_workers=0), 0):
        
        inputs, labels, soft_target = data
        inputs, labels = inputs.to(device), labels.to(device)
        soft_target = soft_target.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss1 = criterion(outputs, labels)
        
        T = 10
        outputs_S = F.log_softmax(outputs/T, dim=1)
        outputs_T = F.softmax(soft_target/T, dim=1)
        
        loss2 = criterion2(outputs_S, outputs_T) * T * T
        
        loss = loss1*(1-alpha) + loss2*alpha
        
        loss.backward()
        optimizer.step()
        
        print('[%d, %5d] loss: %.4f loss1: %.4f loss2: %.4f' %(epoch + 1, (i+1)*batch_size, loss.item(), loss1.item(), loss2.item()))
    
    torch.save(net, 'student2.pkl')
    time_end=time.time()
    print('Time cost:',time_end-time_start, "s")
    
print('Finished Training')

# torch.save(net, 'student2.pkl')
# net = torch.load('student2.pkl')

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





"""
x = torch.tensor([[ 0.2979,  0.0655, -0.0312,  0.0616,  0.0830, 
                   -0.1206, -0.2084, -0.0345,  0.2106, -0.0558]])
y = torch.tensor([5])
print(torch.log(torch.sum(torch.exp(x))) - x[0, y])

criterion = nn.CrossEntropyLoss()
print(criterion(x, y))

"""