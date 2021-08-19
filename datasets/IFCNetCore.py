import os.path as osp
import torch
import pandas as pd

import glob
from torch_geometric.io import read_ply, read_obj
from torch_geometric.data import download_url, extract_zip
import os
import shutil
import numpy as np
# Edited because newest push to master in PytorchGeom is not in pip package (yet)
from helpers.in_memory_dataset import InMemoryDataset
import torch_geometric.transforms as T


class IFCNetCore(InMemoryDataset):
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
        'V1': 'https://ifcnet.e3d.rwth-aachen.de/static/IFCNetCorePly.7z',
        'V2': '..',
    }

    def __init__(self, root, name='Ply', train=True, transform=None, pre_transform=None, pre_filter=None):
        assert name in ['Ply', 'Obj']
        self.name = name

        super(IFCNetCore, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        # helper for loading data from directory, class names must match directory names
        if self.name in ['Ply','Obj']:
            return [
                # all classes in the dataset is in the dataset
                'IfcAirTerminal', 'IfcBeam', 'IfcCableCarrierFitting', 'IfcCableCarrierSegment',
                'IfcDoor', 'IfcDuctFitting', 'IfcDuctSegment', 'IfcFurniture', 'IfcLamp', 'IfcOutlet',
                'IfcPipeFitting', 'IfcPipeSegment', 'IfcPlate', 'IfcRailing', 'IfcSanitaryTerminal', 'IfcSlab',
                'IfcSpaceHeater', 'IfcStair', 'IfcValve', 'IfcWall'
            ]

    @property
    def classmap(self):
        # helper for storing int instead of text for item labels
        if self.name in ['Ply','Obj']:
            return {
                # all classes in the dataset is in the dataset
                1: 'IfcAirTerminal', 2: 'IfcBeam', 3: 'IfcCableCarrierFitting', 4: 'IfcCableCarrierSegment',
                8: 'IfcDoor', 5: 'IfcDuctFitting', 0: 'IfcDuctSegment', 6: 'IfcFurniture', 11: 'IfcLamp', 13: 'IfcOutlet',
                14: 'IfcPipeFitting', 15: 'IfcPipeSegment', 16: 'IfcPlate', 12: 'IfcRailing', 17: 'IfcSanitaryTerminal',
                9: 'IfcSlab', 18: 'IfcSpaceHeater', 7:  'IfcStair', 19: 'IfcValve', 10: 'IfcWall'
            }

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

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
        torch.save(self.process_set('train'), self.processed_paths[0])

        torch.save(self.process_set('test'), self.processed_paths[1])

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
        transform = T.SamplePoints(1024, True, True)
        trans2 = T.GenerateMeshNormals()

        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/*.obj'.format(folder))

            for i, path in enumerate(paths):
                path_list.append(path)

                data = read_obj(path)
                label = list(self.classmap.keys())[list(self.classmap.values()).index(category)]

                try:
                    #trans2(data)
                    #transform(data)

                    a=0
                except:
                    print("failed")
                    continue
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
                    except:
                        print("not all features found skipping this ob")

                    # neighborhood considered or not
                    data.x = torch.cat([x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15], dim=1)
                    # data.x = torch.cat([x14, x15], dim=1)

                    # check consistency
                    assert data.y is not None
                    assert data.x is not None
                data.y = torch.tensor([label])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list2 = []
            for i, d in enumerate(data_list):
                d = self.pre_transform(d)
                data_list2.append(d)

            data_list = data_list2

        return self.collate(data_list)

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))
