import torch
import os
import learning.models
from learning.trainers import Trainer
from torch_geometric.data import DataLoader
import numpy
from sklearn.metrics import confusion_matrix, classification_report
from pandas import DataFrame
from experiments_seg import Experimenter

from prepare_data import indoor3d_util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Inference(Experimenter):
    def __init__(self, path, name, type):
        self.dataset_path = path
        self.dataset_name = name
        self.dataset_type = type

    def infer(self, model_list, test=False, plot = False):
        self.model_list = model_list

        if test:

            for i, model_load in enumerate(self.model_list):
                checkpoint = torch.load(model_load)
                model_name = checkpoint['model']


                if model_name == 'PN2Net_seg':
                    transform, pretransform = self.transform_setup()
                if model_name == 'DGCNNNet_seg':
                    transform, pretransform = self.transform_setup()
                if model_name == 'GUNet_seg':
                    transform, pretransform = self.transform_setup()
                if model_name == 'OWN':
                    transform, pretransform = self.transform_setup()

                # Define datasets
                dataset = self.dataset_name(self.dataset_path, self.dataset_type, True, transform, pretransform)
                test_dataset = self.dataset_name(self.dataset_path, self.dataset_type, False, transform, pretransform)


                test_loader = DataLoader(test_dataset, batch_size=7, shuffle=False, num_workers=8)

                if model_name == 'PN2Net_seg':
                    model_class = getattr(learning.models, model_name)
                    model = model_class(out_channels=test_dataset.num_classes)
                if model_name == 'DGCNNNet_seg':
                    # here we assume test set contains all classes contained in train set...
                    model_class = getattr(learning.models, model_name)
                    model = model_class(out_channels=test_dataset.num_classes)
                if model_name == 'GUNet_seg':
                    model_class = getattr(learning.models, model_name)
                    model = model_class(num_features=test_dataset.num_features, num_classes=test_dataset.num_classes,
                               num_nodes=test_dataset.data.num_nodes).to(device)


                model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.to(device)

                output_path = '/data/output/pred/' + str(i)
                #output_path_error = output_path + '/errors'
                """if not os.path.exists(output_path_error):
                    os.makedirs(output_path_error)"""
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
                real_target_names = [test_dataset.classmap[i] for i in numpy.unique(numpy.array(test_dataset.data.y))]
                class_rep = classification_report(y_true=y_real_list, y_pred=y_pred_list, target_names=real_target_names,
                                                  output_dict=True)
                df2 = DataFrame(class_rep).transpose()
                filename = output_path + '/class_report.csv'
                df2.to_csv(filename)

                """if plot:

                    if model_name == 'PN2Net':
                        vis_point(test_loader, output_path, output_path_error, prob, y_pred_list, y_real_list,
                                  crit_points_list_ind)
                    if model_name == 'DGCNNNet':
                        vis_graph(test_loader, output_path)
                    if model_name == 'UNet':
                        vis_graph(test_loader, output_path)"""


        else:
            print('Warning: inference dataloader not yet implemented')
            return


if __name__ == '__main__':

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_PATH = os.path.join(ROOT_DIR, 'bim-shape-learning/data/room3.txt')
    test = True

    # Name of Dataset, careful the string matters!
    dataset_name = 'S3DIS'  # BIM_PC_T2, ModelNet10, ModelNet40
    config = None
    ex = Experimenter(config, dataset_name)

    experiments = True

    i = 0
    model_list = []
    print(os.getcwd())
    room_path = "data\\input\\room3.txt"
    NUM_POINT = 4096
    current_data, current_label = indoor3d_util.room2blocks_wrapper_normalized(DATA_PATH, NUM_POINT)

    # fills list with all models to perform inference on - scans dir
    while experiments:
        model_path = '/data/output/' + str(i)
        if not os.path.exists(model_path):
            break
        model_list.append(model_path + '/model_state_best_val.pth.tar')
        i += 1

    # fills list with all models to perform inference on - custom models
    #model_list = [] # ['../out/0/model_state_best_val.pth.tar']

    print('inference for {} models with test {}'.format(i, test))

    inf = Inference(ex.dataset_path, ex.dataset_name, ex.dataset_type)

    inf.infer(model_list, test=True, plot=True)
