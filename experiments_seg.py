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

from helpers.results import summary, save_set_stats, save_test_results

# import trainer class
from learning.trainers import Trainer_seg

# Define depending on hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = 'cpu'

NUM_WORKERS = 6
WRITE_DF_TO_ = 'csv', 'latex'



def transform_setup():
    transform = T.KNNGraph(k=2)
    # for now
    # transform = None
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
            output_path_run = os.path.join(output_path, str(i)+"_seg")

            print("Run {} of {}".format(i, len(grid_unfold)))
            print("Writing outputs to {}".format(output_path_run))

            if not os.path.exists(output_path_run):
                os.makedirs(output_path_run)
            if pretrained:
                output_path_run = os.path.join(output_path, str(i), "_seg", "transfer")
                if not os.path.exists(output_path_run):
                    os.makedirs(output_path_run)

            dataset_name_dir, _= dataset_name.split('_')
            self.dataset_path = os.path.join(self.dataset_root_path, dataset_name_dir)
            assert os.path.exists(self.dataset_path)

            if dataset_name[:5] == 'S3DIS':
                self.dataset_name = S3DIS
                self.dataset_type = dataset_name[-1]
            if dataset_name[:6] == 'ASPERN':
                self.dataset_name = ASPERN
                self.dataset_type = dataset_name[-2:]
            else: raise Exception("no dataset_name")

            test_acc, epoch_losses, train_accuracies, val_accuracies, test_ious = self.subrun(output_path_run,
                                                                                              model_name
                                                                                              , n_epochs, batch_size,
                                                                                              learning_rate
                                                                                              , knn, pretrained,
                                                                                              print_set_stats= print_set_stats
                                                                                              , print_model_stats=print_model_stats)
            result['test_acc'] = test_acc
            result['loss'] = epoch_losses
            #result['train_acc'] = train_accuracies
            #result['val_acc'] = val_accuracies
            #result['test_ious'] = test_ious

            results.append(result)

        pd.DataFrame(results).to_csv(os.path.join(output_path, 'results_seg.csv'))
        torch.cuda.empty_cache()

    def subrun(self, output_path_run, model_name, n_epochs, batch_size, learning_rate, knn, pretrained=False,
               print_set_stats=False, print_model_stats=False):

        if model_name.__name__ is 'PN2Net_seg':
            transform, pretransform = transform_setup()
        if model_name.__name__ is 'DGCNNNet_seg':
            transform, pretransform = transform_setup()
        if model_name.__name__ is 'GUNet_seg':
            transform, pretransform = transform_setup()

        # Define datasets (no val dataset)
        train_dataset = self.dataset_name(self.dataset_path, self.dataset_type, True, transform, pretransform)
        test_dataset = self.dataset_name(self.dataset_path, self.dataset_type, False, transform, pretransform)
        val_dataset = self.dataset_name(self.dataset_path, self.dataset_type, False, transform, pretransform)

        print("Run with dataset {} type {}".format(str(self.dataset_name.__name__), str(self.dataset_type)))

        print("Training {} graphs with {} number of classes".format(len(train_dataset), train_dataset.num_classes))
        print("Validating on {} graphs with {} number of classes ".format(len(val_dataset), val_dataset.num_classes))
        print("Testing on {} graphs with {} number of classes ".format(len(test_dataset), test_dataset.num_classes))

        # Define dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

        if print_set_stats:
            # Plots class distributions
            save_set_stats(output_path_run, train_loader, test_loader, train_dataset, val_loader, seg=True)


        if pretrained:
            checkpoint = torch.load(pretrained)
            # say the class output dimension of the pretrained model, for correct loading
            # e.g. if pretrained model was on S3DIS -> set here 13
            dim_last_layer = 13

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

        #Initialize segmentation trainer
        trainer = Trainer_seg(model, output_path_run)
        # Let Trainer run over epochs
        epoch_losses, train_accuracies, val_accuracies = trainer.train(train_loader, val_loader, n_epochs,
                                                                       optimizer)

        test_acc, y_pred, y_real = trainer.test(test_loader, seg=True)
        print("Test accuracy = {}".format(test_acc))

        save_test_results(y_real, y_pred, test_acc, output_path, test_dataset, epoch_losses, train_accuracies,
                          val_accuracies, WRITE_DF_TO_)

        return test_acc, epoch_losses, train_accuracies, val_accuracies, test_ious


if __name__ == '__main__':
    torch.cuda.empty_cache()
    config = dict()

    dataset_root_path = "../.."
    output_path = "../../out_roesti"
    # for Docker use this
    #dataset_root_path = ".."
    #output_path = "/data/output"

    # Name of Dataset, careful the string matters!
    dataset_name = 'ASPERN'  # 'S3DIS'
    print(dataset_name)

    # print set plots
    print_set_stats = True

    # print print model stats
    print_model_stats = True

    # pretrained model?
    pretrained = False
    #pretrained = os.path.join(output_path, "0_seg", "model_state_best_val.pth.tar")

    config['dataset_name'] = ['ASPERN_UG'] #'S3DIS_1' 'ASPERN_UG', ASPERN_DG'
    config['n_epochs'] = [1]
    config['learning_rate'] = [0.001]
    config['batch_size'] = [4]
    config['model_name'] = [DGCNNNet_seg]  # , OWN, PN2Net_seg, DGCNNNet_seg, GUNet_seg
    config['knn'] = [5]
    # config['model_name'] = [, PN2Net, DGCNNNet, , DGCNNNet, UNet]
    ex = Experimenter(config, dataset_root_path, output_path)
    ex.run(print_set_stats, print_model_stats, pretrained)
