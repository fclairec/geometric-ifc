import torch

from torch_geometric.data import Data
from sklearn.model_selection import ParameterGrid
from datasets.s3dis import S3DIS
from datasets.aspern import ASPERN
import torch_geometric.transforms as T
import os
import pandas as pd
from learning.models_seg import DGCNNNet_seg, PN2Net_seg, GUNet_seg, OWN
from datasets.splits import random_splits
from torch_geometric.data import DataLoader
import torch_geometric.nn as nn
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt = matplotlib.pyplot

# for retraining last layer
from torch.nn import Sequential as Seq, Linear as Lin, Dropout
from learning.models import MLP

from helpers.results import summary

# import trainer class
from learning.trainers import Trainer_seg

# Define depending on hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = 'cpu'


class Experimenter(object):
    def __init__(self, config, dataset_name):
        if config is not None:
            self.grid = ParameterGrid(config)
        self.dataset_root_path = '../'
        self.dataset_path = self.dataset_root_path + dataset_name
        if dataset_name[0] == 'S':
            self.dataset_name = S3DIS
            self.dataset_type = 6
        if dataset_name[0] == 'A':
            self.dataset_name = ASPERN
            self.dataset_type = dataset_name[-2:]

    def save_test_results(self, y_real, y_pred, test_acc, output_path, test_dataset, epoch_losses, train_accuracies,
                          val_accuracies):

        # plot epoch losses
        plt.figure()
        plt.plot(range(len(epoch_losses)), epoch_losses, label='training loss')
        plt.legend()
        plt.title("Train loss")
        plt.savefig(output_path + '/train_loss.png')
        plt.close()

        # plot train and val accuracies
        plt.figure()
        plt.plot(range(len(train_accuracies)), train_accuracies, label='training accuracies')
        plt.plot(range(len(train_accuracies)), val_accuracies, label='validation accuracies')
        plt.legend()
        plt.title("Train and validation accuracies")
        plt.savefig(output_path + '/train-val_acc.png')
        plt.close()

    def transform_setup(self):

        transform = T.KNNGraph(k=2)
        # for now
        # transform = None
        pretransform = None

        return transform, pretransform

    def run(self, print_set_stats, print_model_stats, pretrained=False, inference=False):

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

            print("Run {} of {}".format(i, len(grid_unfold)))
            if pretrained:
                output_path = '../data/retrained' + str(i)
            else:
                output_path = '../data' + str(i)


            if not os.path.exists(output_path):
                os.makedirs(output_path)

            test_acc, epoch_losses, train_accuracies, val_accuracies, test_ious = self.subrun(output_path, model_name
                                                                                              , n_epochs, batch_size,
                                                                                              learning_rate
                                                                                              , pretrained,
                                                                                              print_set_stats
                                                                                              , print_model_stats)

            result['test_acc'] = test_acc
            result['loss'] = epoch_losses
            result['train_acc'] = train_accuracies
            result['val_acc'] = val_accuracies
            result['test_ious'] = test_ious

            results.append(result)

        pd.DataFrame(results).to_csv('data/results.csv')
        torch.cuda.empty_cache()

    def subrun(self, output_path, model_name, n_epochs, batch_size, learning_rate, pretrained=False,
               print_set_stats=False, print_model_stats=False):

        if model_name.__name__ is 'PN2Net_seg':
            transform, pretransform = self.transform_setup()
        if model_name.__name__ is 'DGCNNNet_seg':
            transform, pretransform = self.transform_setup()
        if model_name.__name__ is 'GUNet_seg':
            transform, pretransform = self.transform_setup()
        if model_name.__name__ is 'OWN':
            transform, pretransform = self.transform_setup()

        # Define datasets (no val dataset)
        print("Training on {}".format(self.dataset_name.__name__))
        train_dataset = self.dataset_name(self.dataset_path, self.dataset_type, True, transform, pretransform)
        test_dataset = self.dataset_name(self.dataset_path, self.dataset_type, False, transform, pretransform)
        val_dataset = self.dataset_name(self.dataset_path, self.dataset_type, False, transform, pretransform)

        print("Training {} graphs with {} number of classes".format(len(train_dataset), train_dataset.num_classes))
        print("Validating on {} graphs with {} number of classes ".format(len(val_dataset), val_dataset.num_classes))
        print("Testing on {} graphs with {} number of classes ".format(len(test_dataset), test_dataset.num_classes))

        # Define dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        if pretrained:
            checkpoint = torch.load(pretrained)
            # dim_last_layer = checkpoint['num_output_classes']
            # dim last layer of pretrained model, needed for loading
            dim_last_layer = 6

        # Define models depending on the setting
        if model_name.__name__ is 'PN2Net_seg':
            if pretrained:
                # Sets up the model as to fit the old one and to be able to load the weights
                model = model_name(dim_last_layer)
                # Load state
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                # Change last layer to fit new classes of current dataset
                model.mlp = Seq(
                    MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
                    Lin(256, train_dataset.num_classes))
            else:
                model = model_name(train_dataset.num_classes).to(device)
                # model = nn.DataParallel(model)
                # print("Let's use", torch.cuda.device_count(), "GPUs!")

        if model_name.__name__ is 'DGCNNNet_seg':
            if pretrained:
                model = model_name(dim_last_layer)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.mlp = Seq(
                    MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
                    Lin(256, train_dataset.num_classes))
            else:
                model = model_name(out_channels=train_dataset.num_classes)
                # model = nn.DataParallel(model)
                # print("Let's use", torch.cuda.device_count(), "GPUs!")

        if model_name.__name__ is 'GUNet_seg':
            if pretrained:
                model = model_name(num_features=train_dataset.num_features, num_classes=dim_last_layer,
                                   num_nodes=train_dataset.data.num_nodes).to(device)
                model = model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.lin3 = Lin(8, train_dataset.num_classes)
            else:
                model = model_name(num_features=train_dataset.num_features, num_classes=train_dataset.num_classes,
                                   num_nodes=train_dataset.data.num_nodes).to(device)
                # model = nn.DataParallel(model)
                # print("Let's use", torch.cuda.device_count(), "GPUs!")

        model.to(device)

        if print_model_stats:
            summary(model)

        # Define optimizer depending on settings
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.00001)

        trainer = Trainer_seg(model, output_path)
        epoch_losses, train_accuracies, val_accuracies = trainer.train(train_loader, val_loader, n_epochs,
                                                                       optimizer)

        test_acc, y_pred, y_real, test_ious, _ = trainer.test(test_loader)
        print("Test accuracy = {}".format(test_acc))

        self.save_test_results(y_real, y_pred, test_acc, output_path, test_dataset, epoch_losses, train_accuracies,
                               val_accuracies)

        return test_acc, epoch_losses, train_accuracies, val_accuracies, test_ious


if __name__ == '__main__':
    torch.cuda.empty_cache()
    config = dict()

    # Name of Dataset, careful the string matters!
    dataset_name = 'ASPERN' #'S3DIS'
    print(dataset_name)

    # print set plots
    print_set_stats = True

    # print print model stats
    print_model_stats = True

    # pretrained model?
    pretrained = 'data/0/model_state_best_val.pth.tar'

    config['n_epochs'] = [15]
    config['learning_rate'] = [0.001]
    config['batch_size'] = [8]
    config['model_name'] = [DGCNNNet_seg]  # , OWN, PN2Net_seg, DGCNNNet_seg, GUNet_seg
    # config['model_name'] = [, PN2Net, DGCNNNet, , DGCNNNet, UNet]
    ex = Experimenter(config, dataset_name)
    ex.run(print_set_stats, print_model_stats, pretrained)
