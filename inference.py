import torch
from datasets.bim import BIM
from datasets.ModelNet import ModelNet
import os
from learning.models import PN2Net, DGCNNNet, UNet
import learning.models
from learning.trainer import Trainer
from torch_geometric.data import DataLoader
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch_geometric.transforms as T
from sklearn.metrics import confusion_matrix, classification_report
from pandas import DataFrame

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Inference(object):
    def __init__(self, path):
        self.path = path



    def infer(self, test=False):
        if test:



            transform = T.Compose([T.NormalizeScale(), T.SamplePoints(3000)])
            test_data = ModelNet(path, '10', False, transform)
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
            checkpoint = torch.load(model_load)
            model_name = checkpoint['model']

            if model_name == 'PN2Net':
                model_class = getattr(learning.models, model_name)
                model = model_class(out_channels=test_data.num_classes)
            if model_name == 'DGCNNNet':
                # here we assume test set contains all classes contained in train set...
                model_class = getattr(learning.models, model_name)
                model = model_class(out_channels=test_data.num_classes)
            if model_name == 'UNet':
                transform = T.Compose([T.KNNGraph(k=3), T.Distance(), T.Center()])

                dataset = BIM(path, True, transform)
                test_data = BIM(path, False, transform)
                test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=6)
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

            # get labels
            real_target_names = [test_data.classmap[i] for i in numpy.unique(numpy.array(test_data.data.y))]

            # reports
            conf_mat = confusion_matrix(y_true=y_real_list, y_pred=y_pred_list)
            df1 = DataFrame(conf_mat)
            filename = output_path + '/confmat_report.csv'
            df1.to_csv(filename)

            real_target_names = [test_data.classmap[i] for i in numpy.unique(numpy.array(test_data.data.y))]
            class_rep = classification_report(y_true=y_real_list, y_pred=y_pred_list, target_names=real_target_names,
                                              output_dict=True)
            df2 = DataFrame(class_rep).transpose()
            filename = output_path + '/class_report.csv'
            df2.to_csv(filename)






            # Write files
            for i, data in enumerate(test_loader):

                if i==3:
                    break

                #crit_points = [data.pos[j] for j in crit_points_list_ind[i][0]]

                certainty = prob[i]
                y_real = y_real_list[i]
                y_pred = y_pred_list[i]
                y_real_l = test_data.classmap[y_real]
                y_pred_l = test_data.classmap[y_pred]
                pos = data.pos.numpy()

                xyz = numpy.array(
                    [list(a) for a in zip(data.pos[:, 0].numpy(), data.pos[:, 1].numpy(), data.pos[:, 2].numpy())])

                crit_unique_ind = numpy.unique(crit_points_list_ind[i])
                crit_points = numpy.vstack([xyz[j] for j in crit_unique_ind])
                print("Shown {} critical points".format(len(crit_points)))

                ax = plt.axes(projection='3d')
                #ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='black', s=10)
                ax.scatter(crit_points[:, 0], crit_points[:, 1], crit_points[:, 2], color='red', s=20)
                ax.set_xlim(left=1, right=-1)
                ax.set_ylim(bottom=1, top=-1)
                ax.set_zlim(-1, 1)

                if y_pred != y_real:
                    out = output_path_error + "/" + str(i) + y_real_l  + "-" + y_pred_l + "-with(" + str(certainty) + ")"
                    with open(out + ".txt", "w") as text_file:
                        for line in pos:
                            text_file.write(str(line[0]) + ', ' + str(line[1]) + ', ' + str(line[2]) + '\n')
                    plt.savefig(out+'.png')
                    plt.show()
                else:
                    out = output_path + "/" + str(i) + y_real_l  + "-" + y_pred_l + "-with(" + str(certainty) + ")"
                    with open(out + ".txt", "w") as text_file:
                        for line in pos:
                            text_file.write(str(line[0]) + ', ' + str(line[1]) + ', ' + str(line[2]) + '\n')
                    plt.savefig(out + '.png')
                    plt.show()













if __name__ == '__main__':

    test = True

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

    #path = '../../BIM_PC/points'

    path = '../../ModelNet10'
    # if not test set --> change path

    inf = Inference(path)
    inf.infer(test=True)
