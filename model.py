import torch
import torch.nn as nn
import torch.nn.functional as F


class A3C_Labyrinth_Net(nn.Module):
    """
    A3C network with LSTM for 3D maze navigation.
    Architecture from Mnih et al. 2016, Section 8.
    """
    
    def __init__(self, action_space):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, action_space)

    def forward(self, inputs, hidden):
        hx, cx = hidden
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        hx, cx = self.lstm(x, (hx, cx))
        return self.critic_linear(hx), self.actor_linear(hx), (hx, cx)
