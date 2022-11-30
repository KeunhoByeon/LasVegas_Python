import torch
import torch.nn as nn


class PlayerModel(nn.Module):
    def __init__(self, input_num=87, out_num=6):
        super(PlayerModel, self).__init__()

        model = [
            nn.Linear(input_num, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_num),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
