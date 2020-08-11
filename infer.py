import torch
from datasets.bim import BIM
from datasets.ModelNet import ModelNet
import os
from learning.models import PN2Net, DGCNNNet, UNet
import learning.models
from learning.trainers import Trainer_seg
from torch_geometric.data import DataLoader
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch_geometric.transforms as T
from sklearn.metrics import confusion_matrix, classification_report
from pandas import DataFrame
from experiments import Experimenter
from helpers.visualize import vis_point, vis_graph
from datasets.splits import random_splits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Inference(Experimenter):
    def __init__(self, path, name, type):
        self.dataset_path = path
        self.dataset_name = name
        self.dataset_type = type




if __name__ == '__main__':

    test = True

    # Name of Dataset, careful the string matters!
    dataset_name = 'ASPERN_small'  # BIM_PC_T2, ModelNet10, ModelNet40
    config = None
    #ex = Experimenter(config, dataset_name)

    experiments = True

    i = 0
    model_list = []

    # fills list with all models to perform inference on - scans dir
    while experiments:
        model_path = '../out/' + str(i) + "_seg"
        if not os.path.exists(model_path):
            break
        model_list.append(model_path + '/model_state_best_val.pth.tar')
        i += 1

    # fills list with all models to perform inference on - custom models
    #model_list = [] # ['../out/0/model_state_best_val.pth.tar']

    print('inference for {} models with test {}'.format(i, test))

    inf = Inference(dataset_path, ex.dataset_name, ex.dataset_type)

    inf.infer(model_list, test=True, plot=True)
