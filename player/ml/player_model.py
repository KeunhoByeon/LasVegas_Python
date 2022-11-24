import torch
import torch.nn as nn


class PlayerModel(nn.Module):
    def __init__(self, input_num=87, out_num=6, depth_alpha=2, depth_beta=2, train=False, lr=0.0001):
        super(PlayerModel, self).__init__()
        self.training = train
        self.loss = None

        model = []
        for da in range(depth_alpha):
            model.append(nn.Linear(int(input_num * (2 ** da)), int(input_num * (2 ** (da + 1)))))
            model.append(nn.ReLU())
            for db in range(depth_beta):
                model.append(nn.Linear(int(input_num * (2 ** (da + 1))), int(input_num * (2 ** (da + 1)))))
                model.append(nn.ReLU())

        for da in range(depth_alpha):
            model.append(nn.Linear(int(input_num * (2 ** (depth_alpha - da))), int(input_num * (2 ** (depth_alpha - da - 1)))))
            model.append(nn.ReLU())
            for db in range(depth_beta):
                model.append(nn.Linear(int(input_num * (2 ** (depth_alpha - da - 1))), int(input_num * (2 ** (depth_alpha - da - 1)))))
                model.append(nn.ReLU())

        model.append(nn.Linear(input_num, out_num))
        self.model = nn.Sequential(*model)

        if train:
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        else:
            self.model.eval()

    def forward(self, x):
        return self.model(x)
