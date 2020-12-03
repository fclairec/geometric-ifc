import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool,fps, avg_pool_x, voxel_grid, max_pool_x
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, GCNConv
from helpers.graph_unet import GraphUNet
from torch_geometric.utils import dropout_adj
import numpy
from torch.autograd import Variable
from torch_geometric.nn import global_add_pool, global_mean_pool
import torch.nn as nn
#from helpers.glob import global_max_pool, global_add_pool, global_mean_pool



class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class PN2Net(torch.nn.Module):
    def __init__(self, out_channels):
        super(PN2Net, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([6, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, out_channels)

    def forward(self, data):
        sa0_out = (data.norm, data.pos, data.batch)
        #sa0_out = (torch.cat([data.norm, data.pos], dim=1), data.batch)

        #sa0_out = (data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        #x, pos, batch, crit_points = self.sa3_module(*sa2_out)
        x, pos, batch = self.sa3_module(*sa2_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)  , None



class DGCNNNet(torch.nn.Module):
    def __init__(self, out_channels, k=5, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)

        return F.log_softmax(out, dim=1)


class GCNCat(torch.nn.Module):
    def __init__(self, num_classes):
        super(GCNCat, self).__init__()

        self.conv1 = GCNConv(6+2, 64, cached=False, normalize=not True)
        self.conv2 = GCNConv(64+6+2, 128, cached=False, normalize=not True)
        self.conv3 = GCNConv(192+12+4, 256, cached=False, normalize=not True)
        # CAREFUL: If modifying here, check line 202 in experiments.py for pretrained model
        self.lin1 = torch.nn.Linear(256, num_classes)

    def forward(self, data):

        #input = torch.cat([data.norm, data.pos], dim=1)
        #i = torch.cat([data.norm, data.pos, data.x], dim=1)
        input = torch.cat([data.norm, data.pos, data.x], dim=1)
        x, batch = input, data.batch

        edge_index, edge_weight = data.edge_index, data.edge_attr
        edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=edge_index.device)

        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        x = torch.cat([x, input], dim=1)
        x1 = F.dropout(x, training=self.training, p=0.2)
        x1 = F.relu(self.conv2(x1, edge_index, edge_weight))

        x = torch.cat([x, x1, input], dim=1)
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.conv3(x, edge_index, edge_weight))

        out = global_max_pool(x, batch)
        out = self.lin1(out)
        out = F.log_softmax(out, dim=1)
        pred = F.softmax(out, dim=1)
        return out, pred


class GCN(torch.nn.Module):
    def __init__(self, num_classes):
        super(GCN, self).__init__()

        #self.conv1 = GCNConv(6+2, 64, cached=False, normalize=not True)
        #self.conv2 = GCNConv(64+6+2, 128, cached=False, normalize=not True)
        #self.conv3 = GCNConv(128+6+2, 256, cached=False, normalize=not True)
        self.conv1 = GCNConv(6, 64, cached=False, normalize=not True)
        self.conv2 = GCNConv(64+6, 128, cached=False, normalize=not True)
        self.conv3 = GCNConv(128+6, 256, cached=False, normalize=not True)
        # CAREFUL: If modifying here, check line 202 in experiments.py for pretrained model
        self.lin1 = torch.nn.Linear(256, num_classes)

    def forward(self, data):
        #i= torch.cat([data.norm, data.pos, data.x],dim=1)
        #input = torch.cat([data.norm, data.pos, data.x], dim=1)
        i = torch.cat([data.norm, data.pos], dim=1)
        input = torch.cat([data.norm, data.pos],dim=1)
        x, batch = i, data.batch

        edge_index, edge_weight = data.edge_index, data.edge_attr
        edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=edge_index.device)

        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        x = torch.cat([x, input], dim=1)
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        """if batch.size(0) != data.pos.size(0):
            print("nooo")"""

        x = torch.cat([x, input], dim=1)
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        """if batch.size(0) != data.pos.size(0):
            print("nooo")"""

        out = global_max_pool(x, batch)
        out = self.lin1(out)
        out = F.log_softmax(out, dim=1)
        pred = F.softmax(out, dim=1)
        return out, pred

class GCNPool(torch.nn.Module):
    def __init__(self, num_classes):
        super(GCNPool, self).__init__()

        self.conv1 = GCNConv(6+2, 64, cached=False, normalize=not True)

        self.conv2 = GCNConv(64+6+2, 128, cached=False, normalize=not True)

        self.conv3 = GCNConv(128+9+2, 256, cached=False, normalize=not True)
        # CAREFUL: If modifying here, check line 202 in experiments.py for pretrained model
        self.lin1 = torch.nn.Linear(256, num_classes)
        self.con_int = PointConv()

    def forward(self, data):
        #input = torch.cat([data.norm, data.pos], dim=1)
        #i = torch.cat([data.norm, data.pos, data.x], dim=1)
        input = torch.cat([data.norm, data.pos, data.x], dim=1)
        x, batch = input, data.batch

        edge_index, edge_weight = data.edge_index, data.edge_attr
        edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=edge_index.device)

        # first conv with full points
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        # Second conv with full points
        x = torch.cat([x, input], dim=1)
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.relu(self.conv2(x, edge_index, edge_weight))

        #first downsampleing index generation
        idx = fps(data.pos, batch, ratio=0.5)
        row, col = radius(data.pos, data.pos[idx], 0.4, batch, batch[idx], max_num_neighbors=64)
        edge_index_int = torch.stack([col, row], dim=0)
        x = self.con_int(x, (data.pos, data.pos[idx]), edge_index_int)

        batch = batch[idx]

        edge_index, edge_weight = self.filter_adj(edge_index, edge_weight, idx, data.pos.size(0))

        x = torch.cat([x, input[idx]], dim=1)
        x = F.relu(self.conv3(x, edge_index, edge_weight))

        out = global_max_pool(x, batch)
        out = self.lin1(out)
        out = F.log_softmax(out, dim=1)
        pred = F.softmax(out, dim=1)
        return out, pred

    def filter_adj(self, edge_index, edge_attr, perm, num_nodes=None):
        num_nodes = self.maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes,), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_attr is not None:
            edge_attr = edge_attr[mask]

        return torch.stack([row, col], dim=0), edge_attr

    def maybe_num_nodes(self, index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

