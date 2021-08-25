__author__ = 'fiona.collins'

import time
import torch
from sklearn.model_selection import ParameterGrid
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from datasets.BIMGEOM import BIMGEOM
from datasets.IFCNetCore import IFCNetCore
from datasets.ModelNet import ModelNet
from datasets.splits import make_set_sampler

from learning.models import GCNConv
from learning.trainers import Trainer
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from learning.models import MLP
from learning.models import PN2Net, GCNConv, GCNPool, GCNCat, GCN, DGCNNNet, PN2Net

import os
import pandas as pd
import numpy as np
from helpers.results import save_test_results, save_set_stats
from helpers.results import summary
import argparse
import ast
import shutil

NUM_WORKERS = 6

# Define depending on hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("running with:{}".format(device))
# device = 'cpu'

print(os.getcwd())


WRITE_DF_TO_ = ['to_csv']  # , 'to_latex'

def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D Geometric Classifier')
    parser.add_argument('--dataset', default=['BIMGEOMV1', 'IFCNetCoreObj'], nargs='+', type=str, help='dataset name')
    parser.add_argument('--num_epoch', default=[1], nargs='+', type=int, help='number of epochs')
    parser.add_argument('--batch_size', nargs='+', default=[30], type=int, help='batch size')
    parser.add_argument('--learning_rate', nargs='+', default=[0.001], type=float, help='learning rate of optimizer')
    parser.add_argument('--model', default=["GCNPool", "GCNCat", "GCNCat", "PN2Net", "DGCNNNet"], nargs='+', help='model to train') #
    parser.add_argument('--knn', default=[5], nargs='+', help='k nearest point neighbors to connect')
    parser.add_argument('--rotation', default=["[0,0,0]"], nargs='+',
                        help='rotation interval applied to each sample around a specific axis [x, y, z]') #, "[180,180,0]"
    parser.add_argument('--samplePoints', default=[1024], nargs='+', type=int,
                        help='points to sample from mesh surface')
    parser.add_argument('--node_translation', default=[0.0], nargs='+', type=float,
                        help='translation interval applied to each point')
    parser.add_argument('--mesh', default=[False], nargs='+', help='is input a surface mesh?')

    parser.add_argument('--data_path', default='../resources', type=str, help='path to dataset to train')
    parser.add_argument('--output_path', default='../data/Wed_test_inf', type=str, help='output path for experiments')
    parser.add_argument('--logdir', default='./log', type=str, help='path to directory to save log')
    parser.add_argument('--checkpoint_dir', default=False, nargs='+', type=str, help='list of paths to directory to checkpoint')

    args = parser.parse_args()
    return args



def transform_setup(k=5, rotation=[0, 0, 0], samplePoints=1024, mesh=False, node_translation=0.00):
    pre_transform = T.Compose([T.NormalizeScale(), T.Center()])

    # Mesh-to-Graph approach
    if mesh:
        transform = T.Compose([T.RandomRotate(rotation[0], axis=0), T.RandomRotate(rotation[1], axis=1),
                               T.RandomRotate(rotation[2], axis=2),
                               T.GenerateMeshNormals(), T.RandomTranslate(node_translation),
                               T.FaceToEdge(True), T.Distance(norm=True), T.TargetIndegree(cat=True)])

    # Point Sampling & K-nn graph
    else:
        transform = T.Compose([T.RandomRotate(rotation[0], axis=0), T.RandomRotate(rotation[1], axis=1),
                               T.RandomRotate(rotation[2], axis=2),
                               T.GenerateMeshNormals(), T.RandomTranslate(node_translation),
                               T.SamplePoints(samplePoints, True, True), T.KNNGraph(k=k), T.Distance(norm=True)])

    print("The following sample pre-transforms are applied: {}".format(pre_transform))
    print("The following sample transforms are applied: {}".format(transform))

    return transform, pre_transform


class Experimenter(object):
    def __init__(self, config, dataset_root, output_path):

        self.grid = ParameterGrid(config)
        self.dataset_root_path = dataset_root
        self.output_path = output_path



    def run(self, print_set_stats, pretrained, train, inference=False):

        grid_unfold = list(self.grid)
        results = []

        for i, params in enumerate(grid_unfold):

            # set experiment setting
            if not inference:

                dataset_name = params['dataset_name']
                n_epochs = params['n_epochs']
                batch_size = params['batch_size']
                learning_rate = params['learning_rate']
                model_name = params['model_name']
                knn = params['knn']
                rotation = params['rotation']
                sample_points = params['samplePoints']
                mesh = params['mesh']
                node_translation = params['node_translation']
            if inference or pretrained:
                # get setting from the saved model or from a default
                # TODO: save other configs in model info
                self.model_to_load_dir = params['checkpoint_dir']
                info_to_load = pd.read_csv(os.path.join(self.model_to_load_dir, "checkpoint_info.csv"), index_col=0)
                dataset_name = info_to_load["dataset"].values[0]
                model_name = globals()[info_to_load["model"].values[0]]
                n_epochs = 1 # fix
                batch_size = 30 # need this to do efficient testing?
                if inference:
                    self.batch_size_inf = 1
                learning_rate = 0.001
                knn = 5
                rotation = [0,0,0]
                sample_points = 1024
                mesh = False
                node_translation = 0

            # Prepare result output
            result = params
            result['model_name'] = model_name.__name__
            plot_name = ','.join(['%s' % value for (key, value) in result.items()])
            plot_name = plot_name.replace('_', '')

            output_path_run = os.path.join(output_path, str(i) + "_clas")
            print("Run {} of {}".format(i, len(grid_unfold)))
            print("Writing outputs to {}".format(output_path_run))

            if pretrained and train:
                output_path_run = os.path.join(self.model_to_load_dir, "transfer")
                print("transfer learing dir registered")
            if pretrained and not train and inference:
                output_path_run = os.path.join(self.model_to_load_dir, "inference")
                print("inference dir registered")

            if not os.path.exists(output_path_run):
                os.makedirs(output_path_run)

            self.dataset_path = os.path.join(self.dataset_root_path, dataset_name)
            print(os.getcwd())
            print(self.dataset_path)
            """if os.path.exists(os.path.join(self.dataset_path, "processed")):
                shutil.rmtree(os.path.join(self.dataset_path, "processed"))"""

            if dataset_name[0] == 'B':
                self.dataset_name = BIMGEOM
                self.dataset_type = dataset_name[-2:]
            if dataset_name[0] == 'M' and dataset_name[-1] == '0':
                self.dataset_name = ModelNet
                self.dataset_type = dataset_name[-2:]
            if dataset_name[0] == 'I':
                self.dataset_name = IFCNetCore
                self.dataset_type = dataset_name[-3:]

            if print_set_stats:
                # only print set stats once
                set_stats_path = os.path.join(output_path, dataset_name)
                if os.path.exists(set_stats_path):
                    # suppose if path exists we already have stats on the dataset and we skip the analysis from then on
                    print_set_stats = False
                else:
                    os.makedirs(set_stats_path)
                    print_set_stats = set_stats_path
            else:
                print_set_stats = False

            test_acc, epoch_losses, train_accuracies, val_accuracies, epoch_test, mean_epoch_time, num_train_params, num_pos, num_ed = self.subrun(
                output_path_run, n_epochs, model_name, batch_size, learning_rate, knn, pretrained, plot_name, rotation,
                sample_points, node_translation, mesh=mesh, print_set_stats=print_set_stats, train=train, inference=inference)

            result['test_acc'] = test_acc
            result['epoch_test'] = epoch_test
            result['mean_epoch_time'] = mean_epoch_time
            result['num_trainable_parameter'] = num_train_params
            result['num_pos'] = num_pos
            result['num_edge'] = num_ed
            # result['loss'] = epoch_losses
            # result['train_acc'] = train_accuracies
            # result['val_acc'] = val_accuracies

            results.append(result)
            pd.DataFrame(results).to_csv(os.path.join(output_path_run, 'results_clas.csv'))

        pd.DataFrame(results).to_csv(os.path.join(output_path, 'results_clas.csv'))
        torch.cuda.empty_cache()

    def subtrain(self, output_path_run, n_epochs, model, optimizer, train_loader, val_loader, test_loader, test_dataset,
                 plot_name):
        # Initialize Trainer
        trainer = Trainer(model, output_path_run)
        # let Trainer run over epochs
        t0 = time.time()
        epoch_losses, train_accuracies, val_accuracies, epoch_test, mean_epoch_time = trainer.train(train_loader,
                                                                                                    val_loader,
                                                                                                    n_epochs,
                                                                                                    optimizer)
        print('{} seconds'.format(time.time() - t0))
        # Evaluate best model on Test set
        test_acc, y_pred, y_real, _, _, _ = trainer.test(test_loader, seg=False)
        val_acc2, _, _, _, _, _ = trainer.test(val_loader, seg=False)

        print("Test accuracy = {}, Val accuracy = {}".format(test_acc, val_acc2))

        # save test results
        save_test_results(y_real, y_pred, test_acc, output_path_run, test_dataset, epoch_losses, train_accuracies,
                          val_accuracies, WRITE_DF_TO_, plot_name, seg=False)

        return test_acc, epoch_losses, train_accuracies, val_accuracies, epoch_test, mean_epoch_time

    def subrun(self, output_path_run, n_epochs, model_name, batch_size, learning_rate, knn, pretrained=False,
               plot_name=False, rotation=180, sample_points=1024, node_translation=0.001, mesh=False,
               print_set_stats=False, train=True, inference=False):

        if model_name.__name__ is 'PN2Net':
            transform, pre_transform = transform_setup(rotation=rotation, samplePoints=sample_points, mesh=mesh)
        if model_name.__name__ is 'DGCNNNet':
            transform, pre_transform = transform_setup()
        if model_name.__name__ is 'GCN' or 'GCN_cat' or 'GCN_pool' or 'GCNConv':
            # number of knn to connect to as argument
            transform, pre_transform = transform_setup(k=knn, rotation=rotation, samplePoints=sample_points,
                                                       mesh=mesh, node_translation=node_translation)
            if train:
                transform, pre_transform = transform_setup(k=knn, rotation=rotation, samplePoints=sample_points,
                                                           mesh=mesh, node_translation=node_translation)
            else:
                # no need for rotation in inference
                transform, pre_transform = transform_setup(k=knn, rotation=rotation, samplePoints=sample_points,
                                                           mesh=mesh, node_translation=node_translation)

        # Define datasets
        if train:
            dataset = self.dataset_name(self.dataset_path, self.dataset_type, True, transform=transform,
                                        pre_transform=pre_transform)
        test_dataset = self.dataset_name(self.dataset_path, self.dataset_type, False, transform=transform,
                                         pre_transform=pre_transform)

        print("Run with dataset {} type {}".format(str(self.dataset_name.__name__), str(self.dataset_type)))

        # Split dataset randomly (respecting class imbalance) into train and val set (no cross validation for now)
        # _, train_index, val_index = random_split(dataset, dataset.num_classes, train_ratio=0.8)
        if train:
            train_dataset = dataset
            val_dataset = test_dataset  # for now TODO: change to cross validation
            print("Training {} graphs with {} number of classes".format(len(train_dataset), train_dataset.num_classes))
        if inference:
            print("Testing/doing inference on {} graphs with {} number of classes ".format(len(test_dataset), test_dataset.num_classes))

        # Imbalanced datasets: create sampler depending on the length of data per class
        if train:
            sampler_train = make_set_sampler(train_dataset)
        # sampler_val = make_set_sampler(val_dataset)
        if pretrained:
            sampler_test = None
        else:
            sampler_test = make_set_sampler(test_dataset)

        # Define dataloaders
        if train:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUM_WORKERS,
                                      sampler=sampler_train)
            unbalanced_train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
            val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                    sampler=sampler_test)
            # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
            #                        sampler=sampler_val)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                 sampler=sampler_test)
        if inference:
            inf_loader = DataLoader(test_dataset, batch_size=self.batch_size_inf, shuffle=False, num_workers=NUM_WORKERS,
                                 sampler=sampler_test)
        if train:
            num_pos = int(next(iter(train_loader)).pos.size(0) / batch_size)
            num_ed = int(next(iter(train_loader)).edge_index.size(1) / batch_size)
        else:
            num_pos = 0
            num_ed = 0

        if print_set_stats:
            # Plots class distributions
            save_set_stats(print_set_stats, train_loader, test_loader,
                           train_dataset, test_dataset, val_dataset, unbalanced_train_loader, val_loader, seg=False)

        if pretrained:
            # load checkpoints
            pretrained_info = pd.read_csv(os.path.join(self.model_to_load_dir, "checkpoint_info.csv"), index_col=0)
            checkpoint = torch.load(os.path.join(self.model_to_load_dir, "model_state_best_val.pth.tar"), map_location=torch.device('cpu'))
            # say the class output dimension of the pretrained model, for correct loading
            # e.g. if pretrained model was on ModelNet10 -> set here 10
            dim_last_layer = pretrained_info['num_classes'].values[0]

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

        if model_name.__name__ in ['GCN', 'GCNCat', 'GCNPool', 'GCNConv']:
            if pretrained:
                model = model_name(num_classes=dim_last_layer)
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                if train:
                    model.lin1 = Lin(254, test_dataset.num_classes)
                # TODO: Make the loading dynamic
                # model.lin3 = Lin(254, train_dataset.num_classes)
            else:
                model = model_name(num_classes=train_dataset.num_classes).to(device)

        num_trainable_params = summary(model)

        model.to(device)

        # Define optimizer depending on settings
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.00001)
        if pretrained:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if train:
            test_acc, epoch_losses, train_accuracies, val_accuracies, epoch_test, mean_epoch_time = self.subtrain(
                output_path_run, n_epochs, model, optimizer, train_loader, val_loader, test_loader, test_dataset,
                plot_name)

        else:
            # Here inference jobs are done
            #   Class Trainer includes test function so its instanciated here
            trainer = Trainer(model, output_path_run)
            test_acc, _, _, _, _ , _ = trainer.test(test_loader, save_pred=False, seg=False)
            inf_acc, inf_y_pred, inf_y_real, inf_prob, inf_crit_points, final_embedding = trainer.test(inf_loader, save_pred=True, seg=False)

            output_path_error = os.path.join(output_path_run, "error")

            print("overal inference accuracy : {}".format(inf_acc))
            print("overal test accuracy (should be the same.. just different batch size : {}".format(test_acc))
            if not os.path.exists(output_path_error):
                os.makedirs(output_path_error)
            #   TODO: make inference independant of test

            self.inference(test_loader, output_path_run, output_path_error, inf_prob, inf_y_pred, inf_y_real, inf_crit_points, final_embedding)

        # vis_graph(val_loader, output_path)
        # write_pointcloud(val_loader,output_path)$
        if not train:
            epoch_losses = []
            train_accuracies = []
            val_accuracies = []
            epoch_test = []
            mean_epoch_time = []

        return test_acc, epoch_losses, train_accuracies, val_accuracies, epoch_test, mean_epoch_time, num_trainable_params, num_pos, num_ed

    def inference(self, test_loader, output_path_run, output_path_error, prob, y_pred, y_real, crit_points, final_embedding):
        from helpers.visualize import vis_point
        df2 = pd.DataFrame(np.concatenate(final_embedding))
        print(df2)
        df1 = pd.DataFrame(test_loader.dataset.id)

        df_final_embedding = pd.concat([df1, df2], axis=1)


        # save true vs pred with id
        df_truepred = pd.DataFrame({'object_id': test_loader.dataset.id,'y_real': y_real, 'y_pred': y_pred, 'certainty': prob})
        df_truepred.to_csv(os.path.join(output_path_run,"trueVspred-withId.csv"), index=False)


        df_final_embedding.to_csv(os.path.join(output_path_run,"finalembeddings.csv"), index=False)


        #vis_point(test_loader, output_path_run, output_path_error, prob, y_pred, y_real, crit_points)
        return


if __name__ == '__main__':

    args = parse_args()
    #set the following
    mode = "EXP"
    mode = "INF"
    #mode = "TRANS"

    dataset_root_path = args.data_path
    output_path = args.output_path


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # print set plots
    print_set_stats = False

    # for transfer learning we list the pretrained model models here. In case we train from scratch use pretrained_list = [False]
    #pretrained_list = [ "../Resultate/6_out_experiments/3_clas/model_state_best_val.pth.tar", "../Resultate/6_out_experiments/2_clas/model_state_best_val.pth.tar"] # "/data/out_ec3/0_clas/model_state_best_val.pth.tar"
    pretrained_list = [args.checkpoint_dir]


    # Set to false if only performing inference

    config = dict()

    if mode is "EXP":
        train = True
        pretrained = False
        config['dataset_name'] = args.dataset
        config['n_epochs'] = args.num_epoch
        config['learning_rate'] = args.learning_rate
        config['batch_size'] = args.batch_size
        config['model_name'] = [globals()[i] for i in args.model]
        config['knn'] = args.knn  # ,10,15,20
        config['rotation'] = [ast.literal_eval(i) for i in args.rotation]
        config['samplePoints'] = args.samplePoints
        config['node_translation'] = args.node_translation
        config['mesh'] = args.mesh

        # configure experiements
        ex = Experimenter(config, dataset_root_path, output_path)
        # run experiments
        ex.run(print_set_stats, pretrained, train=train)

    if mode is "INF":
        pretrained = True
        train = False
        config['checkpoint_dir'] = args.checkpoint_dir
        ex = Experimenter(config, dataset_root_path, output_path)
        # run experiments
        ex.run(print_set_stats, pretrained, train=train, inference=True)

    if mode is "TRANS":
        pretrained = True
        train = True
        #TODO
        print("to do")

