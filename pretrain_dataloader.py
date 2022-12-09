import os
import random
import sys

import numpy as np
import torch
import torch.utils.data as data
from joblib import Parallel, delayed
from tqdm import tqdm


def load_data_from_csv(csv_path):
    samples = []
    with open(csv_path, 'r') as rf:
        for i, line in enumerate(rf.readlines()):
            line_split = line.replace('\n', '').split(',')
            line_split = np.array(line_split).astype(float)
            input_data, gt = line_split[:-1], int(line_split[-1]) - 1
            samples.append((input_data, gt))

    return samples


def load_data(data_dir, n_process=4):
    csv_paths = []
    for filename in os.listdir(data_dir):
        csv_paths.append(os.path.join(data_dir, filename))

    samples_all = Parallel(n_jobs=n_process)(delayed(load_data_from_csv)(csv_path) for csv_path in tqdm(csv_paths, desc="Loading data", file=sys.stdout))
    samples_all = list(np.concatenate(samples_all))
    random.shuffle(samples_all)

    samples_train = samples_all[: len(samples_all) * 90 // 100]
    samples_test = samples_all[len(samples_all) * 90 // 100:]
    return samples_train, samples_test


class RecordDataset(data.Dataset):
    def __init__(self, data):
        self.samples = data

    def __getitem__(self, index):
        input_data, gt = self.samples[index]
        input_data = torch.FloatTensor([input_data])

        return input_data, gt

    def __len__(self):
        return len(self.samples)
