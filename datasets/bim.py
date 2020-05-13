import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, Data

import glob
from torch_geometric.io import read_txt_array



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



    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):


        super(BIM, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        return [
            'IfcBeam', 'IfcColumn', 'IfcFurnishingElement', 'IfcStairFlight', 'IfcDoor', 'IfcFlowSegment', 'IfcFlowTerminal', 'IfcSlab', 'IfcWallStandardCase'
        ]
    # 'IfcBeam', 'IfcColumn', 'IfcFurnishingElement', 'IfcStairFlight', 'IfcDoor', 'IfcFlowSegment', 'IfcFlowTerminal', 'IfcSlab', 'IfcWallStandardCase'

    @property
    def classmap(self):
        return {0: 'IfcBeam', 1: 'IfcColumn', 2:'IfcFurnishingElement', 3:'IfcStairFlight', 4:'IfcDoor', 5:'IfcFlowSegment', 6:'IfcFlowTerminal', 7:'IfcSlab', 8:'IfcWallStandardCase'}

    #{0: 'IfcBeam', 1: 'IfcColumn', 2:'IfcFurnishingElement', 3:'IfcStairFlight', 4:'IfcDoor', 5:'IfcFlowSegment', 6:'IfcFlowTerminal', 7:'IfcSlab', 8:'IfcWallStandardCase'}

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        return

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):

        #path = osp.join(self.raw_dir, 'pointcloud-tz-test.ply')
        #y_path = osp.join(self.raw_dir, 'ys_long.csv')
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = self.raw_file_names#sorted([x.split(osp.sep)[-2] for x in categories])
        print(categories)

        data_list = []

        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/*.txt'.format(folder))

            for path in paths:
                data = read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None)
                data.y = torch.tensor([target])

                data = Data(y=data.y, pos=data[:, (0, 1, 2)])
                #   data=Data(y=data.y, pos= data[:,(0,1,2)], r= data[:,3],g= data[:,4],b= data[:,5] )

                data_list.append(data)
            self.classmap.update({target:category})




        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)


    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))
