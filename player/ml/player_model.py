import torch
import torch.nn as nn


class PlayerModel(nn.Module):
    def __init__(self, input_num=87, out_num=6, train=False, lr=0.01):
        super(PlayerModel, self).__init__()

        model = [
            nn.Linear(input_num, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, out_num),
        ]
        self.model = nn.Sequential(*model)

        if train:
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=0.01)
            self.scheduler = None
        else:
            self.model.eval()

    def forward(self, x):
        return self.model(x)
