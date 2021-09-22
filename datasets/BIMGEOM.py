import os.path as osp
import torch
import pandas as pd
import pickle

import glob
from torch_geometric.io import read_ply
from torch_geometric.data import download_url, extract_zip
import os
import shutil
import numpy as np
# Edited because newest push to master in PytorchGeom is not in pip package (yet)
from helpers.in_memory_dataset import InMemoryDataset
import torch_geometric.transforms as T


class BIMGEOM(InMemoryDataset):
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
        'V1': 'https://dataverse.harvard.edu/api/access/datafile/4301944',
        'V2': '..',
    }

    def __init__(self, root, name='V1', train=True, transform=None, pre_transform=None, pre_filter=None):
        assert name in ['V1', 'V2']
        self.name = name

        super(BIMGEOM, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        path2 = "train.txt" if train else "test.txt"
        with open(path2, "rb") as fp:  # Unpickling
            self.id = pickle.load(fp)

    @property
    def raw_file_names(self):
        # helper for loading data from directory, class names must match directory names
        if self.name == 'V1':
            return [
                # all classes in the dataset is in the dataset
                'IfcDistributionControlElement', 'IfcFlowController', 'IfcFlowFitting', 'IfcFlowSegment',
                'IfcFlowTerminal', 'IfcColumn', 'IfcFurnishingElement', 'IfcStair', 'IfcDoor', 'IfcSlab',
                'IfcWall', 'IfcWindow', 'IfcRailing'
            ]

    @property
    def classmap(self):
        # helper for storing int instead of text for item labels
        if self.name == 'V1':
            return {
                # all classes in the dataset is in the dataset
                0: 'IfcDistributionControlElement', 1: 'IfcFlowController', 2: 'IfcFlowFitting',
                3: 'IfcFlowSegment', 4: 'IfcFlowTerminal', 5: 'IfcColumn', 6: 'IfcFurnishingElement', 7: 'IfcStair',
                8: 'IfcDoor', 9: 'IfcSlab', 10: 'IfcWall', 11: 'IfcWindow', 12: 'IfcRailing'
            }

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']\



    def download(self):
        path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

        folder = osp.join(self.root, 'BIMGEOM{}'.format(self.name))
        # os.rename(folder, self.raw_dir)

        # Delete osx metadata generated during compression of ModelNet10
        metadata_folder = osp.join(self.root, '__MACOSX')
        if osp.exists(metadata_folder):
            shutil.rmtree(metadata_folder)

    def process(self):
        print(self.processed_paths[0])
        trainers, id_list_train = self.process_set('train')
        torch.save(trainers, self.processed_paths[0])
        with open("train.txt", "wb") as fp:
            pickle.dump(id_list_train, fp)

        testers, id_list_test = self.process_set('test')
        torch.save(testers, self.processed_paths[1])
        with open("test.txt", "wb") as fp:
            pickle.dump(id_list_test, fp)

    def process_set(self, dataset):

        categories = self.raw_file_names
        print("The following classes are processed {}".format(categories))

        # check if a feature file exists for the originating ifc file (e.g. volume, area, neighborhood)
        feature_files = glob.glob(os.path.join(self.raw_dir, '*ifcfeatures_all.csv'))
        print(feature_files)
        if feature_files:
            print("global features considered")
            glob_feat = True
        else:
            print("no global features found")
            glob_feat = False

        all_features = pd.DataFrame()
        for ff in feature_files:
            df = pd.read_csv(ff, index_col=0)
            all_features = all_features.append(df)
        all_features = all_features[~all_features.index.duplicated(keep='first')]

        data_list = []
        path_list = []
        id_list = []

        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/*.ply'.format(folder))

            for i, path in enumerate(paths):
                path_list.append(path)

                data = read_ply(path)
                label = list(self.classmap.keys())[list(self.classmap.values()).index(category)]
                id_list.append(path)

                data.y = torch.tensor([label])



                """trans = T.SamplePoints(1025, True, True)
                trans2 = T.GenerateMeshNormals()
                trans2(data)
                trans(data)
                a=0"""

                if glob_feat:
                    feat_ind = category + "_" + path.split('/')[-1][:22]

                    feat_ind = category + "_" + path.split('/')[-1][:22]
                    try:
                        feature = all_features.loc[feat_ind]
                        # neigh=all_neighbours.loc[feat_ind]
                    except:
                        # print("no features found so not considered")
                        continue

                    try:

                        x1 = torch.tensor([np.log(feature['compactness'] + 1)] * 1024).reshape(-1, 1)
                        
                        x2 = torch.tensor([feature['isWall']] * 1024, dtype=torch.float).reshape(-1, 1)
                        x3 = torch.tensor([feature['isStair']] * 1024, dtype=torch.float).reshape(-1, 1)
                        x4 = torch.tensor([feature['isSlab']] * 1024, dtype=torch.float).reshape(-1, 1)
                        x5 = torch.tensor([feature['isFurn']] * 1024, dtype=torch.float).reshape(-1, 1)
                        x6 = torch.tensor([feature['isCol']] * 1024, dtype=torch.float).reshape(-1, 1)
                        x7 = torch.tensor([feature['isFlowT']] * 1024, dtype=torch.float).reshape(-1, 1)
                        x8 = torch.tensor([feature['isFlowS']] * 1024, dtype=torch.float).reshape(-1, 1)
                        x9 = torch.tensor([feature['isFlowF']] * 1024, dtype=torch.float).reshape(-1, 1)
                        x10 = torch.tensor([feature['isFlowC']] * 1024, dtype=torch.float).reshape(-1, 1)
                        x11 = torch.tensor([feature['isDist']] * 1024, dtype=torch.float).reshape(-1, 1)
                        x12 = torch.tensor([feature['isWin']] * 1024, dtype=torch.float).reshape(-1, 1)
                        x13 = torch.tensor([feature['isDoor']] * 1024, dtype=torch.float).reshape(-1, 1)

                        x14 = torch.tensor([np.log(feature['Volume'] + 1)] * 1024, dtype=torch.float).reshape(-1, 1)
                        x15 = torch.tensor([np.log(feature['Area'] + 1)] * 1024, dtype=torch.float).reshape(-1, 1)
                        print(x2)
                    except:
                        print("not all features found skipping this ob")

                    # neighborhood considered or not
                    data.x = torch.cat([x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15], dim=1)
                    # data.x = torch.cat([x14, x15], dim=1)

                    # check consistency
                    assert data.y is not None
                    assert data.x is not None

                data_list.append(data), id_list


        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list2 = []
            for i, d in enumerate(data_list):
                d = self.pre_transform(d)
                data_list2.append(d)

            data_list = data_list2


        self.id_list = id_list

        return self.collate(data_list), id_list

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))


if __name__ == '__main__':

    print('salman')
