import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, input_num, plane, output_num):
        super(BasicBlock, self).__init__()

        self.fn1 = nn.Linear(input_num, plane)
        self.fn2 = nn.Linear(plane, output_num)
        self.relu = nn.ReLU()
        self.fn_pool = nn.Linear(input_num, output_num)

    def forward(self, x):
        out = self.fn1(x)
        out = self.relu(out)
        out = self.fn2(out) + self.fn_pool(x)
        out = self.relu(out)
        return out


class PlayerModel(nn.Module):
    def __init__(self, input_num=95, output_num=6):
        super(PlayerModel, self).__init__()

        model = [
            nn.Linear(input_num, 256),
            BasicBlock(256, 128, 128),
            BasicBlock(128, 64, 64),
            BasicBlock(64, 32, 32),
            BasicBlock(32, 16, 16),
            nn.Linear(16, output_num)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
