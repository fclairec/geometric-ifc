__author__ = 'fiona.collins'
import time
import torch
from sklearn.model_selection import ParameterGrid
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from datasets.bim import BIM
from datasets.ModelNet import ModelNet
from datasets.ModelNet_small import ModelNet_small
from datasets.splits import random_splits, make_set_sampler

from learning.models import PN2Net, DGCNNNet, GCN, GCNCat, GCNPool
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
#device = 'cpu'


WRITE_DF_TO_ = ['to_csv']#, 'to_latex'


def transform_setup(graph_u=False, graph_gcn=False, rotation=180, samplePoints=1024, mesh=False):
    if not graph_u and not graph_gcn:
        # Default transformation for scale noralization, centering, point sampling and rotating
        pretransform = T.Compose([T.NormalizeScale(), T.Center()])
        transform = T.Compose([T.SamplePoints(samplePoints), T.RandomRotate(rotation)])
        print("pointnet rotation {}" .format(rotation))
    elif graph_u:
        pretransform = T.Compose([T.NormalizeScale(), T.Center()])
        transform = T.Compose([T.NormalizeScale(), T.Center(), T.SamplePoints(samplePoints, True, True), T.RandomRotate(rotation),
                               T.KNNGraph(k=graph_u)])
    elif graph_gcn:

        pretransform = T.Compose([T.NormalizeScale(), T.Center()])

        if mesh:
            if mesh == "extraFeatures":
                transform = T.Compose([T.RandomRotate(rotation), T.GenerateMeshNormals(),
                               T.FaceToEdge(True),  T.Distance(norm=True),T.TargetIndegree(cat=True)]) #,
            else:
                transform = T.Compose([T.RandomRotate(rotation), T.GenerateMeshNormals(),
                                       T.FaceToEdge(True), T.Distance(norm=True), T.TargetIndegree(cat=True)])
        else:
            transform = T.Compose([T.SamplePoints(samplePoints, True, True), T.RandomRotate(rotation),
                               T.KNNGraph(k=graph_gcn), T.Distance(norm=True)])
            print("no mesh")
        print("Rotation {}" . format(rotation))
        print("Meshing {}" . format(mesh))


    else:
        print('no transfom')



    return transform, pretransform


class Experimenter(object):
    def __init__(self, config, dataset_root, output_path):
        if config is not None:
            self.grid = ParameterGrid(config)
        self.dataset_root_path = dataset_root
        self.output_path = output_path

    def run(self, print_set_stats, pretrained=False, inference=False, train=True):

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
            rotation =params['rotation']
            sample_points = params['samplePoints']
            mesh = params['mesh']

            # Prepare result output
            result = params
            result['model_name'] = params['model_name'].__name__
            plot_name = ','.join(['%s' % value for (key, value) in result.items()])
            plot_name=plot_name.replace('_', '')
            # outputpaths
            "assert os.path.exists(output_path)"
            output_path_run = os.path.join(output_path, str(i)+"_clas")

            print("Run {} of {}".format(i, len(grid_unfold)))
            print("Writing outputs to {}".format(output_path_run))

            if pretrained and train:
                output_path_run=os.path.join(os.path.dirname(pretrained), "transfer")
            if pretrained and not train:
                output_path_run = os.path.join(os.path.dirname(pretrained), "inference")
                print("inference")

            if not os.path.exists(output_path_run):
                os.makedirs(output_path_run)



            self.dataset_path = os.path.join(self.dataset_root_path, dataset_name)
            print(os.getcwd())
            print(self.dataset_path)


            #assert os.path.exists(self.dataset_path)


            if dataset_name[0] == 'B':
                self.dataset_name = BIM
                self.dataset_type = dataset_name[-2:]
            if dataset_name[0] == 'M' and dataset_name[-1] == '0':
                self.dataset_name = ModelNet
                self.dataset_type = dataset_name[-2:]
            if dataset_name[0] == 'M' and dataset_name[-1] == 'l':
                self.dataset_name = ModelNet_small
                self.dataset_type = '10'

            if print_set_stats:

                #only print set stats once
                set_stats_path = os.path.join(output_path, dataset_name)
                if os.path.exists(set_stats_path):
                    #suppose if path exists we already have stats on the dataset
                    print_set_stats_run=False

                else:
                    os.makedirs(set_stats_path)
                    print_set_stats_run = set_stats_path
            else: print_set_stats_run=False


            test_acc, epoch_losses, train_accuracies, val_accuracies, epoch_test, mean_epoch_time, num_train_params, num_pos, num_ed = self.subrun( output_path_run, n_epochs, model_name, batch_size, learning_rate, knn, pretrained, plot_name, rotation, sample_points, mesh=mesh, train=train)


            result['test_acc'] = test_acc
            result['epoch_test'] = epoch_test
            result['mean_epoch_time'] = mean_epoch_time
            result['num_trainable_parameter'] = num_train_params
            result['num_pos'] = num_pos
            result['num_edge'] = num_ed
            #result['loss'] = epoch_losses
            # result['train_acc'] = train_accuracies
            # result['val_acc'] = val_accuracies

            results.append(result)
            pd.DataFrame(results).to_csv(os.path.join(output_path_run, 'results_clas.csv'))

        pd.DataFrame(results).to_csv(os.path.join(output_path,'results_clas.csv'))
        torch.cuda.empty_cache()

    def subtrain(self, output_path_run, n_epochs, model, optimizer, train_loader, val_loader, test_loader, test_dataset, plot_name):
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
        test_acc, y_pred, y_real, _ = trainer.test(test_loader, seg=False)
        val_acc2, _, _, _ = trainer.test(val_loader, seg=False)

        print("Test accuracy = {}, Val accuracy = {}".format(test_acc, val_acc2))

        # save test results
        save_test_results(y_real, y_pred, test_acc, output_path_run, test_dataset, epoch_losses, train_accuracies,
                          val_accuracies, WRITE_DF_TO_, plot_name, seg=False)

        return test_acc, epoch_losses, train_accuracies, val_accuracies, epoch_test, mean_epoch_time



    def subrun(self, output_path_run, n_epochs, model_name, batch_size, learning_rate, knn, pretrained=False, plot_name= False, rotation=180, sample_points=1024, mesh=False,
               print_set_stats=False, train=True):

        if model_name.__name__ is 'PN2Net':
            transform, pretransform = transform_setup(rotation=rotation, samplePoints=sample_points, mesh = mesh)
        if model_name.__name__ is 'DGCNNNet':
            transform, pretransform = transform_setup()
        if model_name.__name__ is 'GCN' or 'GCN_cat' or 'GCN_pool':
            # number of knn to connect to as argument
            transform, pretransform = transform_setup(graph_gcn=knn, rotation=rotation, samplePoints=sample_points, mesh = mesh)

        # Define datasets
        dataset = self.dataset_name(self.dataset_path, self.dataset_type, True, transform=transform, pre_transform=pretransform)
        test_dataset = self.dataset_name(self.dataset_path, self.dataset_type, False, transform=transform, pre_transform=pretransform)

        print("Run with dataset {} type {}".format(str(self.dataset_name.__name__), str(self.dataset_type)))

        # Split dataset randomly (respecting class imbalance) into train and val set (no cross validation for now)
        #_, train_index, val_index = random_splits(dataset, dataset.num_classes, train_ratio=0.8)
        train_dataset = dataset
        #train_dataset = dataset[dataset.train_mask].copy_set(train_index)
        #val_dataset = dataset[dataset.val_mask].copy_set(val_index)

        print("Training {} graphs with {} number of classes".format(len(train_dataset), train_dataset.num_classes))
        #print("Validating on {} graphs with {} number of classes ".format(len(val_dataset), val_dataset.num_classes))
        print("Testing on {} graphs with {} number of classes ".format(len(test_dataset), test_dataset.num_classes))

        # Imbalanced datasets: create sampler depending on the length of data per class
        sampler_train = make_set_sampler(train_dataset)
        #sampler_val = make_set_sampler(val_dataset)
        sampler_test = make_set_sampler(test_dataset)

        # Define dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, sampler=sampler_train)
        unbalanced_train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                sampler=sampler_test)
        #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
        #                        sampler=sampler_val)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                 sampler=sampler_test)

        num_pos = int(next(iter(train_loader)).pos.size(0) / batch_size)
        num_ed = int(next(iter(train_loader)).edge_index.size(1)/batch_size)

        if print_set_stats:
            # Plots class distributions
            save_set_stats(print_set_stats, train_loader, test_loader,
                           train_dataset, test_dataset,val_dataset,unbalanced_train_loader, val_loader, seg=False)

        if pretrained:
            checkpoint = torch.load(pretrained)
            # say the class output dimension of the pretrained model, for correct loading
            # e.g. if pretrained model was on ModelNet10 -> set here 10
            dim_last_layer = 5

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

        if model_name.__name__ in ['GCN', 'GCNCat', 'GCNPool']:
            if pretrained:
                model = model_name(num_classes=dim_last_layer)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.lin3 = Lin(254, train_dataset.num_classes)
            else:
                model = model_name(num_classes=dataset.num_classes).to(device)


        num_trainable_params = summary(model)


        model.to(device)

        # Define optimizer depending on settings
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.00001)

        if train:
            test_acc, epoch_losses, train_accuracies, val_accuracies, epoch_test, mean_epoch_time =self.subtrain( output_path_run, n_epochs, model, optimizer, train_loader, val_loader, test_loader, test_dataset, plot_name)

        else:
            trainer = Trainer(model, output_path_run)
            test_acc, y_pred, y_real, prob = trainer.test(test_loader, save_pred=True, seg=False)
            output_path_error = os.path.join(output_path_run, "error")
            self.inference(test_loader,  output_path_run, output_path_error, prob, y_pred, y_real)

        # vis_graph(val_loader, output_path)
        # write_pointcloud(val_loader,output_path)

        return test_acc, epoch_losses, train_accuracies, val_accuracies, epoch_test, mean_epoch_time, num_trainable_params, num_pos, num_ed

    def inference(self, test_loader, output_path_run, output_path_error, prob, y_pred, y_real):
        from helpers.visualize import vis_point, vis_crit_points

        vis_point(test_loader, output_path_run, output_path_error, prob, y_pred, y_real)




if __name__ == '__main__':
    torch.cuda.empty_cache()
    config = dict()

    dataset_root_path = "../"
    output_path = "../out_tmp"
    dataset_root_path = "proj99_tum/"
    output_path = "/data/out_ec3"


    # print set plots
    print_set_stats = False



    # pretrained model
    pretrained = "/tmp/data/0_clas/model_state_best_val.pth.tar" #os.path.join(output_path, "1_clas", "model_state_best_val.pth.tar") (Flase, True or infer)
    # pretrained = False
    # pretrained = os.path.join(output_path, "0_clas", "model_state_best_val.pth.tar")
    train = False #if set to false --> inference


    config['dataset_name'] = ['BIM_PC_C3'] #BIM_PC_T1  #BIM_PC_T4 , 'ModelNet10' 'Benchmark
    config['n_epochs'] = [200]
    config['learning_rate'] = [0.001]
    config['batch_size'] = [25]
    config['model_name'] = [GCN, GCNCat] #GCN GCN_nocat_pool GCN_nocat,GCN, GCN_nocat #, GCN_cat GCN, GCN_cat, GCN_pool, GCN_cat,

    config['knn'] = [5] #,10,15,20
    config['rotation'] = [180]
    config['samplePoints'] = [1024]
    config['mesh'] = [True] # Set to False if KNN #FalseextraFeatures, True, 'extraFeatures', 'extraFeatures'
    # config['model_name'] = [, PN2Net, DGCNNNet, , DGCNNNet, UNetGCN]
    ex = Experimenter(config, dataset_root_path, output_path)
    ex.run(print_set_stats, pretrained, train=train)
