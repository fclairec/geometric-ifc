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


import matplotlib as mpl

plt = mpl.pyplot
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)




from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from pandas import DataFrame

import numpy as np
from helpers.set_plot import Set_analyst
from helpers.visualize import vis_graph, write_pointcloud
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx


# Define depending on hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


class Experimenter(object):
    def __init__(self, config, dataset_name):
        if config is not None:
            self.grid = ParameterGrid(config)
        self.dataset_root_path = '../../'
        self.dataset_path = self.dataset_root_path + dataset_name
        if dataset_name[0] == 'B':
            self.dataset_name = BIM
            self.dataset_type = dataset_name[-2:]
        if dataset_name[0] == 'M' and dataset_name[-1] == '0':
            self.dataset_name = ModelNet
            self.dataset_type = dataset_name[-2:]
        if dataset_name[0] == 'M' and dataset_name[-1] == 'l':
            self.dataset_name = ModelNet_small
            self.dataset_type = '10'

    def transform_setup(self, graph_u=False, graph_gcn=False):
        if not graph_u and not graph_gcn:
            transform = T.Compose([T.NormalizeScale(), T.Center(), T.SamplePoints(1024), T.RandomRotate(180)])
        elif graph_u:
            transform = T.Compose([T.NormalizeScale(), T.Center(), T.RandomRotate(20), T.SamplePoints(1024, True, True), T.KNNGraph(k=5)])
        elif graph_gcn:
            transform = T.Compose([T.NormalizeScale(), T.Center(), T.RandomRotate(180),
                                   T.SamplePoints(1024, True, True), T.KNNGraph(k=5)])
            """T.GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.05),
                  sparsification_kwargs=dict(method='topk', k=10,
                                             dim=0), exact=True)"""
        else: print('no transfom')

        pretransform = None

        return transform, pretransform

    def run(self, print_set_stats, inference=False):

        grid_unfold = list(self.grid)
        results = []

        for i, params in enumerate(grid_unfold):

            # set experiment setting
            n_epochs = params['n_epochs']
            batch_size = params['batch_size']
            learning_rate = params['learning_rate']
            model_name = params['model_name']

            # Prepare result output
            result = params

            print("Run {} of {}" .format(i, len(grid_unfold)))
            output_path = '../out/' + str(i)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            test_acc, epoch_losses, train_accuracies, val_accuracies = self.subrun(output_path, model_name
                                                                                   , n_epochs, batch_size, learning_rate, print_set_stats=print_set_stats)

            result['test_acc'] = test_acc
            result['loss'] = epoch_losses
            result['train_acc'] = train_accuracies
            result['val_acc'] = val_accuracies


            results.append(result)

        pd.DataFrame(results).to_csv('../out/results.csv')
        torch.cuda.empty_cache()

    def subrun(self, output_path, model_name, n_epochs, batch_size, learning_rate, pretrained=False, print_set_stats=False):

        if model_name.__name__ is 'PN2Net':
            transform, pretransform = self.transform_setup()
        if model_name.__name__ is 'DGCNNNet':
            transform, pretransform = self.transform_setup()
        if model_name.__name__ is 'UNet':
            transform, pretransform = self.transform_setup(graph_u=True)
        if model_name.__name__ is 'GCN':
            transform, pretransform = self.transform_setup(graph_gcn=True)

        # Define datasets
        dataset = self.dataset_name(self.dataset_path, self.dataset_type, True, transform, pretransform)
        test_dataset = self.dataset_name(self.dataset_path, self.dataset_type, False, transform, pretransform)

        # maybe here instead of _, dataset is needed
        _, train_index, val_index = random_splits(dataset, dataset.num_classes, train_ratio=0.8)

        train_dataset = dataset[dataset.train_mask].copy_set(train_index)
        val_dataset = dataset[dataset.val_mask].copy_set(val_index)

        print("Training {} graphs with {} number of classes".format(len(train_dataset), train_dataset.num_classes))
        print("Validating on {} graphs with {} number of classes ".format(len(val_dataset), val_dataset.num_classes))
        print("Testing on {} graphs with {} number of classes ".format(len(test_dataset), test_dataset.num_classes))

        # Imbalanced datasets: create sampler depending on the length of data per class
        sampler = make_set_sampler(train_dataset)

        # Define dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=6, sampler=sampler)
        unbalanced_train_loader = DataLoader(train_dataset, batch_size= batch_size, num_workers=6)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

        if print_set_stats:
            self.save_set_stats(train_loader, unbalanced_train_loader, val_loader, test_loader, train_dataset, output_path)

        if pretrained:
            checkpoint=pretrained
            #dim_last_layer = checkpoint['num_output_classes']
            dim_last_layer = 10

        # Define models depending on the setting
        if model_name.__name__ is 'PN2Net':
            if pretrained:
                model = model_name(dim_last_layer)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.mlp = Seq(
                    MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
                    Lin(256, train_dataset.num_classes))
            else: model = model_name(out_channels=train_dataset.num_classes).to(device)

        if model_name.__name__ is 'DGCNNNet':
            if pretrained:
                model = model_name(dim_last_layer)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.mlp = Seq(
                    MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
                    Lin(256, train_dataset.num_classes))
            else: model = model_name(out_channels=train_dataset.num_classes)

        if model_name.__name__ is 'UNet':
            if pretrained:
                model = model_name(num_features=dataset.num_features, num_classes=dim_last_layer,
                               num_nodes=dataset.data.num_nodes).to(device)
                model = model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.lin3 = Lin(8,train_dataset.num_classes)
            else: model = model_name(num_features=dataset.num_features, num_classes=dataset.num_classes,
                               num_nodes=dataset.data.num_nodes).to(device)

        if model_name.__name__ is 'GCN':
            if pretrained:
                model = model_name(num_features=dataset.num_features, num_classes=dim_last_layer,
                               num_nodes=dataset.data.num_nodes).to(device)
                model = model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.lin3 = Lin(8,train_dataset.num_classes)
            else: model = model_name(num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)







        model.to(device)

        # Define optimizer depending on settings
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.00001)

        trainer = Trainer(model, output_path)
        epoch_losses, train_accuracies, val_accuracies = trainer.train(train_loader, val_loader, n_epochs,
                                                                       optimizer)
        test_acc, y_pred, y_real, test_ious, _ = trainer.test(test_loader, seg=False)

        self.save_test_results(y_real, y_pred, test_acc, output_path, test_dataset, epoch_losses, train_accuracies,
                               val_accuracies, test_ious)

        return test_acc, epoch_losses, train_accuracies, val_accuracies



    def save_test_results(self, y_real, y_pred, test_acc, output_path, test_dataset, epoch_losses, train_accuracies, val_accuracies, test_ious):
        df0 = DataFrame(test_ious)
        filename0 = output_path + '/test_ious.csv'
        df0.to_csv(filename0)

        conf_mat = confusion_matrix(y_true=y_real, y_pred=y_pred)
        df1 = DataFrame(conf_mat)
        filename = output_path + '/confmat.csv'
        df1.to_csv(filename)

        target_names = test_dataset.classmap.values()
        real_target_names = [test_dataset.classmap[i] for i in np.unique(np.array(test_dataset.data.y))]
        class_rep = classification_report(y_true=y_real, y_pred=y_pred, target_names=real_target_names,
                                          output_dict=True)
        df2 = DataFrame(class_rep).transpose()
        filename = output_path + '/class_report.csv'
        df2.to_csv(filename)

        print('test acc = {}'.format(test_acc))

        # plot epoch losses
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(range(len(epoch_losses)), epoch_losses, label='training loss', color='steelblue')
        ax.legend()
        ax.set_title("Train loss", fontsize=20)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        ax.set_xticks(range(len(epoch_losses)))
        ax.set_xticklabels(range(len(epoch_losses)), fontsize=12)
        plt.savefig(output_path + '/train_loss.pgf')
        plt.savefig(output_path + '/train_loss.pdf')
        plt.close()

        # plot train and val accuracies
        fig, ax2 = plt.subplots(figsize=(12, 7))
        ax2.plot(range(len(train_accuracies)), train_accuracies, label='Training accuracies', color='steelblue')
        ax2.plot(range(len(train_accuracies)), val_accuracies, label='Validation accuracies', color='indianred')
        ax2.legend(fontsize=16)
        ax2.set_title("Train and validation accuracies", fontsize=20)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Accuracy score', fontsize=16)
        ax2.set_xticks(range(len(train_accuracies)))
        ax2.set_xticklabels(range(len(train_accuracies)), fontsize=12)
        plt.savefig(output_path + '/train-val_acc.pgf')
        plt.savefig(output_path + '/train-val_acc.pdf')
        plt.close()

    def save_set_stats(self, train_loader, unbalanced_train_loader, val_loader, test_loader, train_dataset, output_path):
        Set_analyst(given_set=train_dataset).bar_plot("train_set", output_path)

        Set_analyst([train_loader, unbalanced_train_loader]).bar_plot("train", output_path)
        Set_analyst([val_loader]).bar_plot("val", output_path)
        Set_analyst([test_loader]).bar_plot("test", output_path)

        #vis_graph(val_loader, output_path)
        #write_pointcloud(val_loader,output_path)





if __name__ == '__main__':
    torch.cuda.empty_cache()
    config = dict()

    # Name of Dataset, careful the string matters!
    dataset_name = 'ModelNet10' #BIM_PC_T2, 'BIM_PC_T1', ModelNet10, ModelNet40, ModelNet10_small
    print(dataset_name)

    # print set plots
    print_set_stats = True

    # pretrained model?
    pretrained = False

    config['n_epochs'] = [3]
    config['learning_rate'] = [0.001]
    config['batch_size'] = [8]
    config['model_name'] = [GCN]
    # config['model_name'] = [, PN2Net, DGCNNNet, , DGCNNNet, UNetGCN]
    ex = Experimenter(config, dataset_name)
    ex.run(print_set_stats)



