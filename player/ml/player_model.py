import numpy as np
import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCELoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            raise NotImplementedError('gan mode %s not implemented' % self.gan_mode)

        return loss


class PlayerModel(nn.Module):
    def __init__(self, input_num=87, out_num=6, depth_alpha=1, depth_beta=1, train=False):
        super(PlayerModel, self).__init__()
        self.training = train

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
        model.append(nn.Softmax())
        self.model = nn.Sequential(*model)

        if self.training:
            self.lr = 0.01
            self.criterion = GANLoss(gan_mode='lsgan')
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)  # self.memory = []
            self.true_target = torch.FloatTensor([1.0])
            self.memory = []

    def calc_loss(self, result):
        if result >= 1.0:
            self.loss = self.criterion(torch.FloatTensor(torch.stack(self.memory)), True)
        else:
            self.loss = self.criterion(torch.FloatTensor(torch.stack(self.memory)), False) * (1. - result)
        del self.memory
        self.memory = []

    def optimize_parameters(self, result):
        self.optimizer.zero_grad()
        self.calc_loss(result=result)
        self.loss.backward()
        self.optimizer.step()

    def forward(self, x, return_raw_output=False):
        x = torch.FloatTensor(x)
        if self.training:
            output = self.model(x)
            self.memory.append(output)
        else:
            with torch.no_grad():
                output = self.model(x)

        if return_raw_output:
            return output

        return int(output.topk(1, 1, True, True)[0].item() + 1)


if __name__ == '__main__':
    pm = PlayerModel(depth_alpha=1, depth_beta=1)
    for p_name in pm.state_dict().keys():
        print(p_name, pm.state_dict()[p_name].shape)
