import torch
from helpers.set_plot import Set_analyst
import numpy as np
from torch.utils.data import WeightedRandomSampler


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_splits(dataset, num_classes, train_ratio):
    indices = []
    for i in range(num_classes):
        index = (dataset.data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    l_per_class = Set_analyst(given_set=dataset).class_counter()[0]
    lengths = [tup[1] for tup in list(l_per_class.items())]
    lengths = [int(l * train_ratio) for l in lengths]

    train_index = torch.cat([i[:l] for i, l in zip(indices, lengths)], dim=0)
    rest_index = torch.cat([i[l:] for i, l in zip(indices, lengths)], dim=0)

    print("len indexes")
    print(len(train_index))
    print(len(rest_index))

    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    dataset.train_mask = index_to_mask(train_index, size=len(dataset))
    dataset.val_mask = index_to_mask(rest_index, size=len(dataset))

    return dataset, train_index, rest_index


def make_set_sampler(dataset):

    l_per_class = Set_analyst(given_set=dataset).class_counter()[0]
    print(dataset.num_classes)
    print(l_per_class)
    print(len(dataset.data.y))

    weights_dict = [(tup[0], 1 / tup[1]) for tup in list(l_per_class.items())]
    weights = [1 / tup[1] for tup in list(l_per_class.items())]
    samples_weights = [weights[t] for t in dataset.data.y]
    samples_weights = torch.from_numpy(np.array(samples_weights)).double()
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    return sampler

