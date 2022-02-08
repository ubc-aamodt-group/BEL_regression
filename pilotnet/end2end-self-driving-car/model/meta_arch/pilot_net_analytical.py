import torch
import torch.nn as nn
from model.layer.feed_forward import FeedForward


class PilotNet_ANALYTICAL(nn.Module):
    def __init__(self, cfg, nbits, visualizing=False):
        super(PilotNet_ANALYTICAL, self).__init__()
        self.cfg = cfg
        self.visualizing = visualizing
        self.nbits = nbits

        self.conv1 = nn.Conv2d(3, 24, (5,5), 2)
        self.conv2 = nn.Conv2d(24, 36, (5,5), 2)
        self.conv3 = nn.Conv2d(36, 48, (5,5), 2)
        self.conv4 = nn.Conv2d(48, 64, (3,3), 1)
        self.conv5 = nn.Conv2d(64, 64, (3,3), 1)

        self.act = nn.ELU()

        self.fc1 = nn.Linear(1152, 1000, bias=True)
        self.fc2 = nn.Linear(1000, 500, bias=True)
        self.fc3 = nn.Linear(500, 100, bias=True)
        
        self.to_out = nn.Linear(100, self.nbits)

        torch.nn.init.normal_(self.fc1.weight, mean=1.0, std=0.01)
        torch.nn.init.normal_(self.fc2.weight, mean=1.0, std=0.01)
        torch.nn.init.normal_(self.fc3.weight, mean=1.0, std=0.01)
        torch.nn.init.normal_(self.conv1.weight, mean=1.0, std=0.01)
        torch.nn.init.normal_(self.conv2.weight, mean=1.0, std=0.01)
        torch.nn.init.normal_(self.conv3.weight, mean=1.0, std=0.01)
        torch.nn.init.normal_(self.conv4.weight, mean=1.0, std=0.01)
        torch.nn.init.normal_(self.conv5.weight, mean=1.0, std=0.01)

        # self.feed_forward = nn.Linear(self.cfg.MODEL.FC.INPUT, self.nbits)

        # BUILD LOSS CRITERION
        # self.loss_criterion = nn.MSELoss()

    def forward(self, input, targets=None):
        batch_size = input.size(0)
        normalized_input = input / 127.5 - 1
        print(normalized_input.shape)
        normalized_input = normalized_input.permute(0, 3, 1, 2)   # dislocate depth axis
        print(normalized_input.shape)
        x = self.conv1(normalized_input)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.act(x)
        x = self.conv5(x)
        x = self.act(x)

        x = x.reshape((batch_size, -1))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.to_out(x)

        return x
