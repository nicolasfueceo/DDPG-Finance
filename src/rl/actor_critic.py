import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + shortcut)

class EIIEActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(EIIEActor, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.resblock = ResidualBlock(32)
        conv_output_size = 32 * state_dim[1] * state_dim[2]

        # Include portfolio_weights in the final layers
        self.fc1 = nn.Linear(conv_output_size + action_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state, portfolio_weights):
        if state.dim() == 5:
            state = state.squeeze(1)
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.resblock(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, portfolio_weights), dim=1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

class EIIECritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(EIIECritic, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.resblock = ResidualBlock(32)
        conv_output_size = 32 * state_dim[1] * state_dim[2]

        self.state_fc = nn.Linear(conv_output_size + action_dim, 128)
        self.action_fc = nn.Linear(action_dim, 128)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state, action, portfolio_weights):
        if state.dim() == 5:
            state = state.squeeze(1)
        if action.dim() > 2:
            action = action.squeeze(1)
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.resblock(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, portfolio_weights), dim=1)
        s_out = F.relu(self.state_fc(x))
        a_out = F.relu(self.action_fc(action))
        c = torch.cat([s_out, a_out], dim=1)
        x = F.relu(self.fc1(c))
        return self.fc2(x)
