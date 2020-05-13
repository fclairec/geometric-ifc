__author__ = 'fiona.collins'

import torch
from sklearn.model_selection import ParameterGrid
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from torch_cluster import knn_graph

from datasets.bim import BIM
from datasets.ModelNet import ModelNet

from learning.models import PN2Net, DGCNNNet, UNet
from learning.trainer import Trainer

import os
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import inspect
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from pandas import DataFrame
from numpy import concatenate as concat
import numpy as np
from helpers.set_plot import set_analyst
from sklearn.model_selection import StratifiedKFold



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'


class Experimenter(object):
    def __init__(self, config):
        self.grid = ParameterGrid(config)

    def run(self):
        #path = '../data/Dummy'
        cdw= cwd = os.getcwd()

        transform = T.Compose([T.NormalizeScale(), T.Center()])

        path = '../../BIM_PC_small/points'
        dataset = BIM(path, True, transform)
        test_data = BIM(path, False, transform)



        """
        path = '../../ModelNet40'
        transform, pre_transform = T.NormalizeScale(), T.SamplePoints(1024)
        dataset = ModelNet(path, '40', True, transform, pre_transform)
        test_data = ModelNet(path, '40', False, transform, pre_transform)
        """

        num_graphs = len(dataset)
        train_size = int(0.8 * num_graphs)
        val_size = int(0.2 * num_graphs)
        print("Training graphs: ", train_size)
        print("Validation graphs: ", val_size)

        train_data = dataset[:train_size]
        val_data = dataset[train_size:train_size + val_size]

        results = []
        grid_unfold = list(self.grid)

        for i, params in enumerate(grid_unfold):
            print("Run {} of {}".format(i, len(grid_unfold)))
            output_path = '../out/' + str(i)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            result = params

            n_epochs = params['n_epochs']
            batch_size = params['batch_size']
            learning_rate = params['learning_rate']
            model_name = params['model_name']

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
            set_analyst(train_data, 'train_data')
            set_analyst(test_data,'test_data')
            set_analyst(val_data, 'val_data')

            if model_name.__name__ is 'PN2Net':
                model = model_name(out_channels=train_data.num_classes).to(device)
            if model_name.__name__ is 'DGCNNNet':
                model = model_name(out_channels=train_data.num_classes).to(device)

            if model_name.__name__ is 'UNet':
                # TODO : make a bit nicer...
                transform = T.Compose([T.KNNGraph(k=3), T.NormalizeScale(), T.Center()])
                dataset = BIM(path, True, transform)
                test_data = BIM(path, False, transform)
                train_data = dataset[:train_size]
                val_data = dataset[train_size:train_size + val_size]
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
                val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
                test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
                model = model_name(num_features=dataset.num_features, num_classes=dataset.num_classes, num_nodes=dataset.data.num_nodes).to(device)
            #num_features, num_classes, num_nodes, edge_index

            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.00001)

            trainer = Trainer(model, output_path)  # ,max_patience=5
            epoch_losses, train_accuracies, val_accuracies = trainer.train(train_loader, val_loader, n_epochs, optimizer)

            #threshold = trainer.optim_threshold(val_loader)
            #result['threshold'] = threshold

            test_acc, y_pred, y_real = trainer.test(test_loader)
            result['test_acc'] = test_acc

            conf_mat = confusion_matrix(y_true=y_real, y_pred=y_pred)
            df1 = DataFrame(conf_mat)
            filename = output_path + '/confmat_report.csv'
            df1.to_csv(filename)

            target_names = test_data.classmap.values()
            real_target_names = [test_data.classmap[i] for i in np.unique(np.array(dataset.data.y))]
            class_rep = classification_report(y_true=y_real, y_pred=y_pred, target_names=real_target_names, output_dict=True)
            df2 = DataFrame(class_rep).transpose()
            filename = output_path + '/class_report.csv'
            df2.to_csv(filename)



            print('test acc = {}' .format(test_acc))

            plt.figure()
            plt.plot(range(len(epoch_losses)), epoch_losses, label='training loss')
            plt.legend()
            plt.savefig(output_path + '/training.png')
            plt.close()

            a = plt
            a.figure()
            a.plot(range(len(train_accuracies)), train_accuracies, label='training accuracies')
            a.plot(range(len(train_accuracies)), val_accuracies, label='validation accuracies')
            a.legend()
            a.savefig(output_path + '/training_acc.png')
            a.close()

            results.append(result)

            #self.plot_latent_space(model, train_loader)

            """sampler = Sampler(model)
            num_nodes = 10
            num_graphs = 2
            score = sampler.sample(num_nodes, num_graphs, threshold, enc_output_dim, dataset.id2type, dataset.id2rel, output_path)
            result['score'] = score"""
        # TODO: add final trainings loss, validato?
        pd.DataFrame(results).to_csv('../out/results.csv')
        torch.cuda.empty_cache()



if __name__ == '__main__':
    torch.cuda.empty_cache()
    config = dict()

    config['n_epochs'] = [20]
    config['learning_rate'] = [1e-2]
    config['batch_size'] = [8]
    config['model_name'] = [UNet,PN2Net, DGCNNNet]
    #config['model_name'] = [, PN2Net, DGCNNNet]
    ex = Experimenter(config)
    ex.run()


