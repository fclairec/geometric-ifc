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
from shutil import copyfile

# for retraining last layer
from torch.nn import Sequential as Seq, Linear as Lin, Dropout
from learning.models import MLP
from torch_geometric.nn import GCNConv

from helpers.results import summary, save_set_stats, save_test_results

# import trainer class
from learning.trainers import Trainer_seg

# Define depending on hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = 'cpu'

NUM_WORKERS = 8
WRITE_DF_TO_ = 'csv', 'latex'

model_dict = {"DGCNNNet_seg": DGCNNNet_seg, "GUNet_seg": GUNet_seg, "PN2Net_seg": PN2Net_seg}


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

    def run(self, print_set_stats, print_model_stats, pretrained=False):

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
            result['model_name'] = params['model_name'].__name__
            plot_name = ','.join(['%s' % value for (key, value) in result.items()])
            plot_name = plot_name.replace('_', '')

            # outputpaths
            assert os.path.exists(output_path)
            output_path_run = os.path.join(output_path, str(i) + '_seg')

            if not os.path.exists(output_path_run):
                os.makedirs(output_path_run)
            if pretrained:
                output_path_run = os.path.join(output_path, str(i) + "_seg" + "_transfer")
                if not os.path.exists(output_path_run):
                    os.makedirs(output_path_run)

            print("Run {} of {}".format(i, len(grid_unfold)))
            print("Writing outputs to {}".format(output_path_run))

            dataset_name_dir, _ = dataset_name.split('_')
            self.dataset_path = os.path.join(self.dataset_root_path, dataset_name_dir)
            # assert os.path.exists(self.dataset_path)

            if dataset_name[:5] == 'S3DIS':
                self.dataset_name = S3DIS
                self.dataset_type = dataset_name[-1]
            elif dataset_name[:6] == 'ASPERN' and dataset_name[-4:] == 'full':
                self.dataset_name = ASPERN
                self.dataset_type = 'full'
            elif dataset_name[:6] == 'ASPERN' and dataset_name[-5:] == 'small':
                self.dataset_name = ASPERN
                self.dataset_type = 'small'
            else:
                raise Exception("no dataset_name")

            test_acc, epoch_losses, train_accuracies, val_accuracies, epoch_test = self.subrun(output_path_run,
                                                                                   model_name
                                                                                   , n_epochs, batch_size,
                                                                                   learning_rate
                                                                                   , knn, pretrained,  plot_name= plot_name,
                                                                                   print_set_stats=print_set_stats
                                                                                   , print_model_stats=print_model_stats
                                                                                   , inference=False)
            result['test_acc'] = test_acc
            result['epoch_test'] = epoch_test
            # result['loss'] = epoch_losses
            # result['train_acc'] = train_accuracies
            # result['val_acc'] = val_accuracies
            # result['test_ious'] = test_ious

            results.append(result)

        pd.DataFrame(results).to_csv(os.path.join(output_path, 'results_seg.csv'))
        torch.cuda.empty_cache()

    def inference(self, pretrained, inference, dataset_name, dataset_path, dataset_type):

        self.dataset_path = dataset_path
        self.dataset_type = dataset_type

        dataset_name_dir, _ = dataset_name.split('_')
        self.dataset_path = os.path.join(self.dataset_root_path, dataset_name_dir)

        if dataset_name[:5] == 'S3DIS':
            self.dataset_name = S3DIS
            self.dataset_type = dataset_name[-1]
        elif dataset_name[:6] == 'ASPERN' and dataset_name[-4:] == 'full':
            self.dataset_name = ASPERN
            self.dataset_type = dataset_name[-2:]
        elif dataset_name[:6] == 'ASPERN' and dataset_name[-5:] == 'small':
            self.dataset_name = ASPERN
            self.dataset_type = 'small'
            self.dataset_path = self.dataset_path + "_small"
        else:
            raise Exception("no dataset_name")


        output_path = self.output_path

        for i, model_i in enumerate(pretrained):
            checkpoint = torch.load(model_i)
            model_name = checkpoint['model']
            model_name = model_dict[model_name]

            n_epochs = 1
            batch_size = 2
            learning_rate = 0.001
            knn = 5

            prediction_path = os.path.join(output_path, str(i) + "_seg"+ "_predictions")
            if not os.path.exists(prediction_path):
                os.makedirs(prediction_path)
            plot_name = model_name.__name__ +',inference'
            plot_name = plot_name.replace('_', '')

            test_acc, epoch_losses, train_accuracies, val_accuracies, epoch_test = self.subrun(prediction_path,
                                                                               model_name
                                                                               , n_epochs, batch_size,
                                                                               learning_rate
                                                                               , knn, pretrained=model_i,  plot_name= plot_name,
                                                                               print_set_stats=print_set_stats
                                                                               , print_model_stats=print_model_stats
                                                                               , inference=True, save_pred=True, prediction_path=prediction_path)

        return

    def subrun(self, output_path_run, model_name, n_epochs, batch_size, learning_rate, knn, pretrained=False,  plot_name= False,
               print_set_stats=False, print_model_stats=False, inference=False, save_pred=False, prediction_path=None):

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
        if inference:
            inf_dataset = self.dataset_name(self.dataset_path, self.dataset_type, 'nonnorm', transform, pretransform)

        print("Run with dataset {} type {}".format(str(self.dataset_name.__name__), str(self.dataset_type)))

        print("Training {} graphs with {} number of classes".format(len(train_dataset), train_dataset.num_classes))
        print("Validating on {} graphs with {} number of classes ".format(len(val_dataset), val_dataset.num_classes))
        print("Testing on {} graphs with {} number of classes ".format(len(test_dataset), test_dataset.num_classes))

        # Define dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        if inference:
            inf_loader = DataLoader(inf_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        else:
            inf_loader = None

        if print_set_stats:
            # Plots class distributions
            save_set_stats(output_path_run, train_loader, test_loader, train_dataset, val_loader, seg=True)

        if pretrained:
            checkpoint = torch.load(pretrained)
            best_performance = checkpoint['best_train_acc']
            # say the class output dimension of the pretrained model, for correct loading
            # e.g. if pretrained model was on S3DIS -> set here 13
            dim_last_layer = 13
            print("retaining on new dataset with previous last dim layer {}" .format(dim_last_layer))
            #copy pretrained model over, in case, even after retraining the accuracy doesn't beat the old model
            if inference:
                copyfile(pretrained, os.path.join(prediction_path, 'model_state_best_val.pth.tar'))
            else:
                # no inference but retrain model
                copyfile(pretrained, os.path.join(output_path_run, 'model_state_best_val.pth.tar'))
                # set best performance to zero because previsou model is irrelevant
                best_performance = 0
        else: best_performance = 0


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
                # Define optimizer depending on settings
                optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.00001)
                optimizer.load_state_dict(checkpoint['optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            else:
                model = model_name(train_dataset.num_classes).to(device)
                optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.00001)
                # model = nn.DataParallel(model)
                # print("Let's use", torch.cuda.device_count(), "GPUs!")

        if model_name.__name__ is 'DGCNNNet_seg':
            if pretrained:
                model = model_name(dim_last_layer)
                model.load_state_dict(checkpoint['state_dict'], strict=False)

                optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.00001)
                optimizer.load_state_dict(checkpoint['optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
                model.mlp = Seq(
                    MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
                    Lin(256, train_dataset.num_classes))
            else:
                model = model_name(out_channels=train_dataset.num_classes)
                optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.00001)
                # model = nn.DataParallel(model)
                # print("Let's use", torch.cuda.device_count(), "GPUs!")

        if model_name.__name__ is 'GUNet_seg':
            if pretrained:
                model = model_name(num_features=train_dataset.num_features, num_classes=dim_last_layer,
                                   num_nodes=train_dataset.data.num_nodes).to(device)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.unet.up_convs[-1] = GCNConv(128, train_dataset.num_classes, improved=True)
                #model.lin3 = Lin(8, train_dataset.num_classes)
                optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.00001)
                optimizer.load_state_dict(checkpoint['optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            else:
                model = model_name(num_features=train_dataset.num_features, num_classes=train_dataset.num_classes,
                                   num_nodes=train_dataset.data.num_nodes).to(device)
                optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.00001)
                # model = nn.DataParallel(model)
                # print("Let's use", torch.cuda.device_count(), "GPUs!")

        if pretrained:
            #we need to rename the copied base for transfer learning
            os.rename(os.path.join(output_path_run, 'model_state_best_val.pth.tar'), os.path.join(output_path_run, 'model_state_best_val_basemodel.pth.tar'))
        if inference:
            os.rename(os.path.join(prediction_path, 'model_state_best_val.pth.tar'), os.path.join(prediction_path, 'model_state_best_val_basemodel.pth.tar'))

        model.to(device)

        if print_model_stats:
            summary(model)



        # Initialize segmentation trainer
        trainer = Trainer_seg(model, output_path_run)
        # Let Trainer run over epochs
        epoch_losses, train_accuracies, val_accuracies, epoch_test = trainer.train(train_loader, val_loader, n_epochs,
                                                                       optimizer, best_performance=best_performance)

        test_acc, y_pred, y_real = trainer.test(test_loader, seg=True, save_pred=save_pred, prediction_path=prediction_path, inf_dataloader=inf_loader)
        print("Test accuracy = {}".format(test_acc))

        #save_test_results(y_real, y_pred, test_acc, output_path_run, test_dataset, epoch_losses, train_accuracies,
        #                  val_accuracies, WRITE_DF_TO_, plot_name, seg=True)

        return test_acc, epoch_losses, train_accuracies, val_accuracies, epoch_test


if __name__ == '__main__':
    torch.cuda.empty_cache()
    config = dict()

    # for roesti computer use this
    #dataset_root_path = "../.."
    #output_path = "../../out_roesti"
    # for Docker use this
    dataset_root_path = ""
    output_path = "/data/output"

    # Name of Dataset, careful the string matters!

    # print set plots
    print_set_stats = False

    # print print model stats
    print_model_stats = False

    # pretrained model?
    # pretrained = False
    # inference = False
    # os.path.join(output_path, "0_seg", "model_state_best_val.pth.tar")
    pretrained = os.path.join(output_path, "2_seg", "model_state_best_val.pth.tar")

    inference = False

    config['dataset_name'] = ['ASPERN_full']  # 'S3DIS_1' 'ASPERN_UG', ASPERN_DG' 'ASPERN_small' , ASPERN_full
    config['n_epochs'] = [80]
    config['learning_rate'] = [0.001]
    config['batch_size'] = [10]
    config['model_name'] = [DGCNNNet_seg]  # , OWN, PN2Net_seg, DGCNNNet_seg, GUNet_seg
    config['knn'] = [5]
    # config['model_name'] = [, PN2Net, DGCNNNet, , DGCNNNet, UNet]
    ex = Experimenter(config, dataset_root_path, output_path)

    if not inference:
        print("training...")
        ex.run(print_set_stats, print_model_stats, pretrained)
    if inference:
        dataset_path = os.path.join(dataset_root_path, 'ASPERN_full')
        dataset_type = 'small'
        ex.inference(pretrained, inference, dataset_name='ASPERN_full', dataset_path=dataset_path, dataset_type=dataset_type)
