import multiprocessing as mp
import math
import argparse
import os
import random
import sys
import time
from time import strftime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from game_manager import GameManager
from logger import Logger
from player.ml.player_model import PlayerModel
from player.ml_player import MLPlayer
from pretrain_dataloader import RecordDataset, load_data


def test_game(model, logger=None):
    PLAYER_SETTING = ["MLPlayer", "Random", "Random", "Random", "Random"]

    total_ranking = [[0 for _ in range(len(PLAYER_SETTING))] for _ in range(len(PLAYER_SETTING) + 1)]
    total_ranking_sum = [0 for _ in range(len(PLAYER_SETTING) + 1)]
    game_manager = GameManager(print_game=False)

    for slot_index, player_type in enumerate(PLAYER_SETTING):
        game_manager.add_player(slot_index=slot_index + 1, player_type=player_type)

    if isinstance(game_manager.players_manager._player_slots[0], MLPlayer):
        game_manager.players_manager._player_slots[0].model.load_state_dict(model.state_dict())

    for _ in range(1000):
        ranking = game_manager.run()
        for rank, player_index in enumerate(ranking):
            total_ranking[player_index][rank] += 1
            total_ranking_sum[player_index] += rank

    logger('Player{} {} ({})'.format(1, total_ranking[1], total_ranking_sum[1:]))


def val(args, epoch, model, criterion, val_loader, logger=None):
    model.eval()

    accuracy = [0, 0]
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), leave=False, desc='Validation {}'.format(epoch), total=len(val_loader), file=sys.stdout):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            output = output[:, 0, :]
            pred = F.softmax(output, dim=1)
            loss = criterion(pred, target)

            acc = torch.sum(torch.argmax(pred.detach(), dim=1) == target).item() / len(pred)
            accuracy[0] += torch.sum(torch.argmax(pred.detach(), dim=1) == target).item()
            accuracy[1] += len(input)
            logger.add_history('val', {"loss": loss.item(), "acc": acc})

        if logger is not None:
            logger('*Validation', history_key='val', time=strftime('%Y-%m-%d %I:%M:%S %p', time.localtime()))

    return accuracy[0] / accuracy[1]


def train(args, epoch, model, criterion, optimizer, train_loader, logger=None):
    model.train()

    num_progress, next_print = 0, args.print_freq
    for i, (input, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        output = output[:, 0, :]
        pred = F.softmax(output, dim=1)

        optimizer.zero_grad()
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        acc = torch.sum(torch.argmax(pred.detach(), dim=1) == target).item() / len(pred)

        logger.add_history('batch', {"loss": loss.item(), "acc": acc})
        logger.add_history('total', {"loss": loss.item(), "acc": acc})

        num_progress += len(input)
        if num_progress >= next_print:
            if logger is not None:
                logger(history_key='batch', epoch=epoch, batch=num_progress, lr=round(optimizer.param_groups[0]['lr'], 6), time=strftime('%Y-%m-%d %I:%M:%S %p', time.localtime()))
            next_print += args.print_freq

    if logger is not None:
        logger(history_key='total', epoch=epoch)


def run(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    model = PlayerModel()

    if args.resume is not None:
        state_dict = torch.load(args.resume, map_location='cpu')
        try:
            model.load_state_dict(state_dict)
        except:
            model = state_dict

    if torch.cuda.is_available():
        model = model.cuda()

    train_set, test_set = load_data(args.data, n_process=args.workers)
    print("Train: {}  Test: {}".format(len(train_set), len(test_set)))
    train_dataset = RecordDataset(train_set)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    val_dataset = RecordDataset(test_set)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    del train_set, test_set

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    if args.schedule is not None:
        need_cycle = math.ceil(math.log(args.schedule, 2))
        step_size = args.epochs / need_cycle
        step_size_up = int(step_size / 10)
        step_size_down = int(step_size / 10 * 9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=args.lr, base_lr=args.lr / args.schedule, step_size_up=step_size_up, step_size_down=step_size_down, mode='triangular2', cycle_momentum=False)
        scheduler.step(step_size_up)
        print('step_size_up, step_size_down', step_size_up, step_size_down)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader.dataset), float_round=6)
    logger.set_sort(['acc', 'loss', 'lr', 'time'])
    logger(str(args))

    best_acc = 0
    for epoch in range(args.epochs):
        train(args, epoch, model, criterion, optimizer, train_loader, logger=logger)
        acc = val(args, epoch, model, criterion, val_loader, logger=logger)
        if args.schedule is not None:
            scheduler.step()
        # test_game(model, logger=logger)

        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.join(args.result, 'checkpoints'), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.result, 'checkpoints', '{}.pth'.format(epoch)))
            torch.save(model, os.path.join(args.result, 'model_best.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=1024 * 128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--seed', default=103, type=int)
    parser.add_argument('--schedule', default=None, type=int)
    parser.add_argument('--data', default='./data')
    parser.add_argument('--result', default='./results', help='path to results')
    parser.add_argument('--resume', default=None, help='path to checkpoint')
    parser.add_argument('--print_freq', default=1000000, type=int, help='print frequency (default: 1000)')
    args = parser.parse_args()

    args.result = os.path.join(args.result, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    os.makedirs(args.result, exist_ok=True)

    run(args)
