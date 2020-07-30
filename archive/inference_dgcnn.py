import torch
from datasets.bim import BIM
from datasets.ModelNet import ModelNet
import os
from learning.models import PN2Net, DGCNNNet, UNet
import learning.models
from learning.trainers import Trainer
from torch_geometric.data import DataLoader
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch_geometric.transforms as T
from sklearn.metrics import confusion_matrix, classification_report
from pandas import DataFrame
from helpers.visualize import vis_point, vis_graph
from helpers.set_plot import Set_analyst

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Inference(object):
    def __init__(self, path, plot):
        self.path = path

    def infer(self, test=False):
        if test:

            transform = T.Compose([T.SamplePoints(1024), T.NormalizeScale(), T.Center(), T.KNNGraph(k=5)])
            dataset = BIM(self.path, 'T1', True, transform)

            transform = T.Compose([T.SamplePoints(1024), T.NormalizeScale(), T.Center(), T.KNNGraph(k=5)])

            path = '../../../ModelNet10'
            dataset = ModelNet(path, '10', True, transform)
            l_per_class = Set_analyst(given_set=dataset).class_counter()


            test_data = BIM(self.path, 'T1', False, transform)
            #test_data=test_data[:20].copy_set()
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)


            """
            transform = T.Compose([T.Distance(), T.Center()])

            test_data = BIM(path, False, transform)
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=6)
            """
        else:
            print('Warning: inference dataloader not yet implemented')
            return

        for i, model_load in enumerate(model_list):
            output_path = '../out/pred/' + str(1)
            vis_graph(test_loader, output_path)

            checkpoint = torch.load(model_load)
            model_name = checkpoint['model']

            if model_name == 'PN2Net':
                model_class = getattr(learning.models, model_name)
                model = model_class(out_channels=dataset.num_classes)
            if model_name == 'DGCNNNet':
                # here we assume test set contains all classes contained in train set...
                model_class = getattr(learning.models, model_name)
                model = model_class(out_channels=dataset.num_classes)
            if model_name == 'UNet':
                transform = T.Compose([T.KNNGraph(k=3), T.Distance(), T.Center()])

                dataset = ModelNet(path, '10', True, transform)
                test_data = ModelNet(path, '10', False, transform)
                #test_data = test_data[:15].copy_set()

                test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
                model_class = getattr(learning.models, model_name)
                model = model_class(num_features=dataset.num_features, num_classes=dataset.num_classes,
                                    num_nodes=dataset.data.num_nodes).to(device)

            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.to(device)

            output_path = '../out/pred/' + str(i)
            output_path_error = output_path + '/errors'
            if not os.path.exists(output_path_error):
                os.makedirs(output_path_error)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            trainer = Trainer(model, output_path)
            test_acc, y_pred_list, y_real_list, prob, crit_points_list_ind = trainer.test(test_loader)
            print(test_acc)

            # reports
            conf_mat = confusion_matrix(y_true=y_real_list, y_pred=y_pred_list)
            df1 = DataFrame(conf_mat)
            filename = output_path + '/confmat_report.csv'
            df1.to_csv(filename)

            # get labels
            real_target_names = [test_data.classmap[i] for i in numpy.unique(numpy.array(test_data.data.y))]
            class_rep = classification_report(y_true=y_real_list, y_pred=y_pred_list, target_names=real_target_names,
                                              output_dict=True)
            df2 = DataFrame(class_rep).transpose()
            filename = output_path + '/class_report.csv'
            df2.to_csv(filename)

            if plot:
                vis_point(test_loader, output_path, output_path_error, prob, y_pred_list, y_real_list,
                          crit_points_list_ind)


if __name__ == '__main__':

    test = True
    # print plot?
    plot = True

    experiments = True
    i = 0
    model_list = []
    while experiments:
        model_path = '../out/' + str(i)
        if not os.path.exists(model_path):
            break
        model_list.append(model_path + '/model_state_best_val.pth.tar')
        i += 1
    print('inference for {} models with test {}'.format(i, test))

    # path = '../../BIM_PC/points'

    path = '../../BIM_PC_T1'
    # if not test set --> change path

    inf = Inference(path, plot)
    inf.infer(test=True)
