import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchsummary import summary


class ResNet(nn.Module):
    # ResNet for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000,num_bits=10,code="u"):
        print(block.expansion)
        print(layers)
        code_bits={'u':[200,200],'j':[100,100],'b1jdj':[51,51],'b2jdj':[27,27],'hex16':[16,16]}
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        if num_bits>0:
            self.fc_angles_yaw = nn.Sequential( nn.Linear(512 * block.expansion,num_bits),nn.Linear(num_bits,1))
            self.fc_angles_pitch = nn.Sequential( nn.Linear(512 * block.expansion,num_bits),nn.Linear(num_bits,1))
            self.fc_angles_roll = nn.Sequential( nn.Linear(512 * block.expansion,num_bits),nn.Linear(num_bits,1))
        else:
            self.fc_angles_yaw = nn.Sequential( nn.Linear(512 * block.expansion,1))
            self.fc_angles_pitch = nn.Sequential( nn.Linear(512 * block.expansion,1))
            self.fc_angles_roll = nn.Sequential( nn.Linear(512 * block.expansion,1))

        self.yawm = nn.Sequential( nn.Linear(512 * block.expansion,1))
        self.pitchm = nn.Sequential( nn.Linear(512 * block.expansion,1))
        self.rollm = nn.Sequential( nn.Linear(512 * block.expansion,1))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        yaw = self.fc_angles_yaw(x)


        pitch = self.fc_angles_pitch(x)

        roll = self.fc_angles_roll(x)

        return yaw,pitch,roll,1 #torch.cat((y,p,r),dim=1)
