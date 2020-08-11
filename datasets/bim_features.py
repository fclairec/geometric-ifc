import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, Data

import glob
from torch_geometric.io import read_txt_array, read_ply
from torch_geometric.data import download_url, extract_zip
import os
import shutil
# Edited because newest push to master in PytorchGeom is not in pip package (yet)
from helpers.in_memory_dataset import InMemoryDataset


class BIM(InMemoryDataset):
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
        'T1':  'https://drive.google.com/uc?export=download&id=1YrlVZpjbpxAMWswPa3gPlNMEPu4nKH8Z',
        'T2': 'https://drive.google.com/uc?export=download&id=1uAbELJqkCCuB01iAbWDx8wZECaUZiGh1',
        'T3': 'https://drive.google.com/uc?export=download&id=1uAbELJqkCCuB01iAbWDx8wZECaUZiGh1'

    }

    def __init__(self, root, name='T1', train=True, transform=None, pre_transform=None, pre_filter=None):
        assert name in ['T1', 'T2', 'T3']
        self.name = name

        super(BIM, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
            if self.name == 'T1':
                return [
                    # corresponds to what is in the dataset
                    'IfcColumn', 'IfcFurnishingElement', 'IfcStair', 'IfcDoor', 'IfcSlab', 'IfcWall', 'IfcWindow'
                ]
            elif self.name == 'T2':
                return [
                    # corresponds to what is in the dataset
                    'IfcDistributionControlElement', 'IfcFlowController', 'IfcFlowFitting', 'IfcFlowSegment',
                    'IfcFlowTerminal'
                ]
            elif self.name == 'T3':
                return [
                    # corresponds to what is in the dataset
                    'IfcDistributionControlElement', 'IfcFlowController', 'IfcFlowFitting', 'IfcFlowSegment',
                    'IfcFlowTerminal', 'IfcColumn', 'IfcFurnishingElement', 'IfcStair', 'IfcDoor', 'IfcSlab', 'IfcWall'
                    , 'IfcWindow'
                ]

    @property
    def classmap(self):
        if self.name == 'T1':
            return {
                # corresponds to what is in the dataset
                0: 'IfcColumn', 1: 'IfcFurnishingElement', 2: 'IfcStair', 3: 'IfcDoor', 4: 'IfcSlab', 5: 'IfcWall',
                    6: 'IfcWindow'
            }
        elif self.name == 'T2':
            return {
                # corresponds to what is in the dataset
                0: 'IfcDistributionControlElement', 1: 'IfcFlowController', 2: 'IfcFlowFitting',
                    3: 'IfcFlowSegment', 4: 'IfcFlowTerminal'
            }
        elif self.name == 'T3':
            return {
                # corresponds to what is in the dataset
                0: 'IfcDistributionControlElement', 1: 'IfcFlowController', 2: 'IfcFlowFitting',
                    3: 'IfcFlowSegment', 4: 'IfcFlowTerminal', 5: 'IfcColumn', 6: 'IfcFurnishingElement', 7: 'IfcStair',
                    8: 'IfcDoor', 9: 'IfcSlab', 10: 'IfcWall', 11: 'IfcWindow'
            }

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):

        path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        folder = osp.join(self.root, 'BIM_PC_{}'.format(self.name))
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)

        # Delete osx metadata generated during compression of ModelNet10
        metadata_folder = osp.join(self.root, '__MACOSX')
        if osp.exists(metadata_folder):
            shutil.rmtree(metadata_folder)

        return

    def process(self):
        print(self.processed_paths[0])
        torch.save(self.process_set('train'), self.processed_paths[0])

        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):

        categories = self.raw_file_names
        print(categories)

        data_list = []

        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/*.ply'.format(folder))
            feature_paths = glob.glob('{}/*.txt'.format(folder))

            for path, feature_path in zip(paths, feature_paths):
                data = read_ply(path)
                data.y = torch.tensor([target])



                data_list.append(data)
            self.classmap.update({target:category})

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))
