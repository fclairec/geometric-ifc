import os
import os.path as osp
import shutil

import h5py
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)


class ASPERN(InMemoryDataset):
    r"""The (pre-processed) Stanford Large-Scale 3D Indoor Spaces dataset from
    the `"3D Semantic Parsing of Large-Scale Indoor Spaces"
    <http://buildingparser.stanford.edu/images/3D_Semantic_Parsing.pdf>`_
    paper, containing point clouds of six large-scale indoor parts in three
    buildings with 12 semantic elements (and one clutter class).

    Args:
        root (string): Root directory where the dataset should be saved.
        test_area (int, optional): Which area to use for testing (1-6).
            (default: :obj:`6`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = ('https://shapenet.cs.stanford.edu/media/'
           'indoor3d_sem_seg_hdf5_data.zip')

    def __init__(self,
                 root,
                 test_area='OG',
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.test_area = test_area
        super(ASPERN, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['all_files_train.txt', 'all_files_test.txt']

    @property
    def classmap(self):
        return {0: 'pipes', 1: 'walls', 2: 'slabs', 3: 'stairs', 4: 'doors', 5: 'column', 6: 'tanks'
                }

    @property
    def processed_file_names(self):
        test_area = self.test_area
        return ['{}_{}.pt'.format(s, test_area) for s in ['train', 'test']]

    def download(self):
        return

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            filenames = [x.split('/')[-1] for x in f.read().split('\n')[:-1]]
        with open(self.raw_paths[1], 'r') as f:
            filenames_test = [x.split('/')[-1] for x in f.read().split('\n')[:-1]]

        xs, ys = [], []
        for filename in filenames:
            f = h5py.File(osp.join(self.raw_dir, filename))
            xs += torch.from_numpy(f['data'][:]).unbind(0)
            ys += torch.from_numpy(f['label'][:]).to(torch.long).unbind(0)


        train_data_list = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            data = Data(pos=x[:, :3], x=x[:, 3:], y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            train_data_list.append(data)

        xs_test, ys_test = [], []
        for filename in filenames_test:
            f = h5py.File(osp.join(self.raw_dir, filename))
            xs_test += torch.from_numpy(f['data'][:]).unbind(0)
            ys_test += torch.from_numpy(f['label'][:]).to(torch.long).unbind(0)

        test_data_list = []
        for i, (x, y) in enumerate(zip(xs_test, ys_test)):
            data = Data(pos=x[:, :3], x=x[:, 3:], y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            test_data_list.append(data)

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])
