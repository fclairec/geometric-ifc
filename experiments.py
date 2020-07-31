__author__ = 'fiona.collins'

import torch
from sklearn.model_selection import ParameterGrid
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from datasets.bim import BIM
from datasets.ModelNet import ModelNet
from datasets.ModelNet_small import ModelNet_small
from datasets.splits import random_splits, make_set_sampler

from learning.models import PN2Net, DGCNNNet, UNet, PointNet, GCN
from learning.trainers import Trainer
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from learning.models import MLP

import os
import pandas as pd

from helpers.results import save_test_results, save_set_stats
from helpers.results import summary

from helpers.visualize import vis_graph, write_pointcloud

# Define depending on hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

NUM_WORKERS = 6
WRITE_DF_TO_ = ['to_csv', 'to_latex']


def transform_setup(graph_u=False, graph_gcn=False):
    if not graph_u and not graph_gcn:
        # Default transformation for scale noralization, centering, point sampling and rotating
        transform = T.Compose([T.NormalizeScale(), T.Center(), T.SamplePoints(1024), T.RandomRotate(180)])
    elif graph_u:
        transform = T.Compose([T.NormalizeScale(), T.Center(), T.SamplePoints(1024, True, True), T.RandomRotate(180),
                               T.KNNGraph(k=graph_u)])
    elif graph_gcn:
        transform = T.Compose([T.NormalizeScale(), T.Center(), T.SamplePoints(1024, True, True), T.RandomRotate(180),
                               T.KNNGraph(k=graph_gcn)])
        """T.GDC(self_loop_weight=1, normalization_in='sym',
              normalization_out='col',
              diffusion_kwargs=dict(method='ppr', alpha=0.05),
              sparsification_kwargs=dict(method='topk', k=10,
                                         dim=0), exact=True)"""
    else:
        print('no transfom')

    pretransform = None

    return transform, pretransform


class Experimenter(object):
    def __init__(self, config, dataset_root, output_path):
        if config is not None:
            self.grid = ParameterGrid(config)
        self.dataset_root_path = dataset_root
        self.output_path = output_path

    def run(self, print_set_stats, print_model_stats, pretrained=False, inference=False):

        grid_unfold = list(self.grid)
        results = []

        for i, params in enumerate(grid_unfold):

            # set experiment setting
            dataset_name = params['dataset_name']
            n_epochs = params['n_epochs']
            batch_size = params['batch_size']
            learning_rate = params['learning_rate']
            model_name = params['model_name']
            knn = params['knn']

            # Prepare result output
            result = params

            # outputpaths
            assert os.path.exists(output_path)
            output_path_run = os.path.join(output_path, str(i)+"_clas")

            print("Run {} of {}".format(i, len(grid_unfold)))
            print("Writing outputs to {}".format(output_path_run))

            if not os.path.exists(output_path_run):
                os.makedirs(output_path_run)
            if pretrained:
                output_path_run = os.path.join(output_path, str(i), "_clas", "transfer")
                if not os.path.exists(output_path_run):
                    os.makedirs(output_path_run)

            self.dataset_path = os.path.join(self.dataset_root_path, dataset_name)
            assert os.path.exists(self.dataset_path)

            if dataset_name[0] == 'B':
                self.dataset_name = BIM
                self.dataset_type = dataset_name[-2:]
            if dataset_name[0] == 'M' and dataset_name[-1] == '0':
                self.dataset_name = ModelNet
                self.dataset_type = dataset_name[-2:]
            if dataset_name[0] == 'M' and dataset_name[-1] == 'l':
                self.dataset_name = ModelNet_small
                self.dataset_type = '10'

            test_acc, epoch_losses, train_accuracies, val_accuracies = self.subrun(output_path_run, model_name
                                                                                   , n_epochs, batch_size,
                                                                                   learning_rate, knn, pretrained,
                                                                                   print_set_stats=print_set_stats, print_model_stats=print_model_stats)

            result['test_acc'] = test_acc
            result['loss'] = epoch_losses
            # result['train_acc'] = train_accuracies
            # result['val_acc'] = val_accuracies

            results.append(result)

        pd.DataFrame(results).to_csv(os.path.join(output_path,'results_clas.csv'))
        torch.cuda.empty_cache()

    def subrun(self, output_path_run, model_name, n_epochs, batch_size, learning_rate, knn, pretrained=False,
               print_set_stats=False, print_model_stats=False):

        if model_name.__name__ is 'PN2Net':
            transform, pretransform = transform_setup()
        if model_name.__name__ is 'DGCNNNet':
            transform, pretransform = transform_setup()
        if model_name.__name__ is 'GCN':
            # number of knn to connect to as argument
            transform, pretransform = transform_setup(graph_gcn=knn)

        # Define datasets
        dataset = self.dataset_name(self.dataset_path, self.dataset_type, True, transform, pretransform)
        test_dataset = self.dataset_name(self.dataset_path, self.dataset_type, False, transform, pretransform)

        print("Run with dataset {} type {}".format(str(self.dataset_name.__name__), str(self.dataset_type)))

        # Split dataset randomly (respecting class imbalance) into train and val set (no cross validation for now)
        _, train_index, val_index = random_splits(dataset, dataset.num_classes, train_ratio=0.8)
        train_dataset = dataset[dataset.train_mask].copy_set(train_index)
        val_dataset = dataset[dataset.val_mask].copy_set(val_index)

        print("Training {} graphs with {} number of classes".format(len(train_dataset), train_dataset.num_classes))
        print("Validating on {} graphs with {} number of classes ".format(len(val_dataset), val_dataset.num_classes))
        print("Testing on {} graphs with {} number of classes ".format(len(test_dataset), test_dataset.num_classes))

        # Imbalanced datasets: create sampler depending on the length of data per class
        sampler_train = make_set_sampler(train_dataset)
        sampler_val = make_set_sampler(val_dataset)
        sampler_test = make_set_sampler(test_dataset)

        # Define dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, sampler=sampler_train)
        unbalanced_train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                sampler=sampler_val)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                 sampler=sampler_test)

        if print_set_stats:
            # Plots class distributions
            save_set_stats(output_path_run, train_loader, unbalanced_train_loader, val_loader, test_loader,
                           train_dataset)

        if pretrained:
            checkpoint = torch.load(pretrained)
            # say the class output dimension of the pretrained model, for correct loading
            # e.g. if pretrained model was on ModelNet10 -> set here 10
            dim_last_layer = 10

        # Define models depending on the setting
        if model_name.__name__ is 'PN2Net':
            if pretrained:
                model = model_name(dim_last_layer)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.mlp = Seq(
                    MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
                    Lin(256, train_dataset.num_classes))
            else:
                model = model_name(out_channels=train_dataset.num_classes).to(device)

        if model_name.__name__ is 'DGCNNNet':
            if pretrained:
                model = model_name(dim_last_layer)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.mlp = Seq(
                    MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
                    Lin(256, train_dataset.num_classes))
            else:
                model = model_name(out_channels=train_dataset.num_classes)

        if model_name.__name__ is 'GCN':
            if pretrained:
                model = model_name(num_features=dataset.num_features, num_classes=dim_last_layer,
                                   num_nodes=dataset.data.num_nodes).to(device)
                model = model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.lin3 = Lin(254, train_dataset.num_classes)
            else:
                model = model_name(num_classes=dataset.num_classes).to(device)

        if print_model_stats:
            summary(model)

        model.to(device)

        # Define optimizer depending on settings
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.00001)

        # Initialize Trainer
        trainer = Trainer(model, output_path_run)
        # let Trainer run over epochs
        epoch_losses, train_accuracies, val_accuracies = trainer.train(train_loader, val_loader, n_epochs,
                                                                       optimizer)
        # Evaluate best model on Test set
        test_acc, y_pred, y_real = trainer.test(test_loader, seg=False)

        print("Test accuracy = {}".format(test_acc))

        # save test results
        save_test_results(y_real, y_pred, test_acc, output_path_run, test_dataset, epoch_losses, train_accuracies,
                          val_accuracies, WRITE_DF_TO_, seg=False)

        # vis_graph(val_loader, output_path)
        # write_pointcloud(val_loader,output_path)

        return test_acc, epoch_losses, train_accuracies, val_accuracies


if __name__ == '__main__':
    torch.cuda.empty_cache()
    config = dict()

    dataset_root_path = "../.."
    output_path = "../../out_roesti"

    # print set plots
    print_set_stats = True

    # print model stats
    print_model_stats = True

    # pretrained model
    pretrained = False
    # pretrained = os.path.join(output_path, "0_clas", "model_state_best_val.pth.tar")

    config['dataset_name'] = ['ModelNet10']
    config['n_epochs'] = [1]
    config['learning_rate'] = [0.001]
    config['batch_size'] = [8]
    config['model_name'] = [GCN] #GCN
    config['knn'] = [5]
    # config['model_name'] = [, PN2Net, DGCNNNet, , DGCNNNet, UNetGCN]
    ex = Experimenter(config, dataset_root_path, output_path)
    ex.run(print_set_stats, print_model_stats, pretrained)
