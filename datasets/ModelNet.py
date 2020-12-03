import os
import os.path as osp
import shutil
import glob

import torch
from torch_geometric.data import download_url, extract_zip, Dataset, InMemoryDataset
from torch_geometric.io import read_off
from helpers.in_memory_dataset import InMemoryDataset


class ModelNet(InMemoryDataset):  #
    r"""The ModelNet10/40 datasets from the `"3D ShapeNets: A Deep
    Representation for Volumetric Shapes"
    <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
    containing CAD models of 10 and 40 categories, respectively.
    .. note::
        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): The name of the dataset (:obj:`"10"` for
            ModelNet10, :obj:`"40"` for ModelNet40). (default: :obj:`"10"`)
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

    urls = {
        '10':
            'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip',
        '40': 'http://modelnet.cs.princeton.edu/ModelNet40.zip'

    }

    # '40': 'https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip'

    def __init__(self, root, name='10', features=None, train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        assert name in ['10', '40']
        self.name = name
        super(ModelNet, self).__init__(root, transform, pre_transform,
                                       pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door',
            'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard','lamp', 'laptop', 'mantel', 'monitor',
            'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent',
            'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'

        ]


    #   'tv_stand'
    # 'airplane', 'bottle',  'cone', 'bowl', 'bathtub',+'car','cup', 'laptop','flower_pot', 'glass_box', 'guitar', 'keyboard', 'mantel', 'person','piano', 'plant', 'radio', 'range_hood','tent', 'vase', 'xbox'

    @property
    def classmap(self):
        return {0: 'airplane', 1: 'bathtub', 2: 'bed', 3:'bench', 4:'bookshelf', 5:'bottle', 6:'bowl', 7:'car',8: 'chair',
                9: 'cone', 10: 'cup', 11:'curtain', 12: 'desk',13: 'door', 14: 'dresser', 15: 'flower_pot', 16: 'glass_box',
                17: 'guitar', 18:'keyboard', 19:'lamp', 20:'laptop', 21:'mantel',22: 'monitor', 23: 'night_stand', 24:'person',
                25:'piano', 26:'plant', 27:'radio', 28:'range_hood', 29:'sink', 30: 'sofa', 31:'stairs', 32:'stool',
                33: 'table', 34: 'tent', 35: 'toilet', 36:'tv_stand', 37:'vase', 38:'wardrobe', 39:'xbox'
                }

    """
    #   18: 'tv_stand'

    #10:'airplane', 14:'bottle', 17:'cone',16:'laptop', 16:'car', 13:'bowl', 18:'cup', 20: 'flower_pot',  9:'bathtub',22:'guitar', 23:'keyboard', 26:'mantel',29:'person', 30: 'piano', 31: 'plant', 32: 'radio', 33: 'range_hood', 38: 'vase', 39: 'xbox', 36: 'tent',21:'glass_box',
    """

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):

        path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        folder = osp.join(self.root, 'ModelNet{}'.format(self.name))
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)

        # Delete osx metadata generated during compression of ModelNet10
        metadata_folder = osp.join(self.root, '__MACOSX')
        if osp.exists(metadata_folder):
            shutil.rmtree(metadata_folder)

        return

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = self.raw_file_names  # sorted([x.split(os.sep)[-2] for x in categories])

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/{}_*.off'.format(folder, category))
            for path in paths:
                data = read_off(path)
                data.y = torch.tensor([target])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))