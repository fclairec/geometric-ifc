import os.path as osp
from sklearn.metrics import accuracy_score
import torch
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib as mpl
import numpy as np
from helpers.set_plot import Set_analyst

plt = mpl.pyplot
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)


class Results():
    def __init__(self):
        self.train_losses = []
        self.val_accuracy = []
        # self.val_aps = []
        self.val_rocs = []
        # self.test_aps = []
        # self.test_rocs = []
        self.correct = 0

    def update(self, train_loss=None, val_accuracy=None, val_roc=None):
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_accuracy is not None:
            self.val_accuracy

    def updateValAccuracy(self, label, pred, nb):
        acc = 0
        self.correct += pred.eq(label).sum().item()
        self.val_accuracy.append(self.correct / nb)


def summary(model):
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)


def calculate_sem_IoU(self, pred_np, seg_np, num_classes):
    print(num_classes)
    I_all = np.zeros(num_classes)
    U_all = np.zeros(num_classes)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(num_classes):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all


def save_test_results(y_real, y_pred, test_acc, output_path, test_dataset, epoch_losses, train_accuracies,
                      val_accuracies, WRITE_DF_TO_, seg=False):
    if seg:
        test_ious = calculate_sem_IoU(y_pred, y_real, test_dataset.num_classes)

        # write Test IOUS
        df0 = DataFrame(test_ious)
        file_ious_csv = output_path + '/test_ious.csv'
        df0.to_csv(file_ious_csv)
        if 'to_latex' in WRITE_DF_TO_:
            file_ious_tex = output_path + '/test_ious.tex'
            df0.to_latex(file_ious_tex, index=False)

    if not seg:
        real_target_names = [test_dataset.classmap[i] for i in np.unique(np.array(test_dataset.data.y))]

        # Confusion matrix
        conf_mat = confusion_matrix(y_true=y_real, y_pred=y_pred)
        df1 = DataFrame(conf_mat, index=real_target_names, columns=real_target_names)
        file_confmat_csv = output_path + '/confmat.csv'
        df1.to_csv(file_confmat_csv)
        if 'to_latex' in WRITE_DF_TO_:
            file_confmat_tex = output_path + '/confmat.tex'
            df1.to_latex(file_confmat_tex)
            print("latex written")

        # Classification report
        class_rep = classification_report(y_true=y_real, y_pred=y_pred, target_names=real_target_names,
                                          output_dict=True)
        df2 = DataFrame(class_rep).transpose()
        file_classrep_csv = output_path + '/class_report.csv'
        df2.to_csv(file_classrep_csv)
        if 'to_latex' in WRITE_DF_TO_:
            file_classrep_tex = output_path + '/class_report.tex'
            df2.to_latex(file_classrep_tex)

    print('test acc = {}'.format(test_acc))

    # plot epoch losses
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(range(len(epoch_losses)), epoch_losses, label='training loss', color='steelblue')
    ax.legend()
    ax.set_title("Train loss", fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    ax.set_xticks(range(len(epoch_losses)))
    ax.set_xticklabels(range(len(epoch_losses)), fontsize=12)
    plt.savefig(output_path + '/train_loss.pgf')
    plt.savefig(output_path + '/train_loss.pdf')
    plt.close()

    # plot train and val accuracies
    fig, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(range(len(train_accuracies)), train_accuracies, label='Training accuracies', color='steelblue')
    ax2.plot(range(len(train_accuracies)), val_accuracies, label='Validation accuracies', color='indianred')
    ax2.legend(fontsize=16)
    ax2.set_title("Train and validation accuracies", fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy score', fontsize=16)
    ax2.set_xticks(range(len(train_accuracies)))
    ax2.set_xticklabels(range(len(train_accuracies)), fontsize=12)
    plt.savefig(output_path + '/train-val_acc.pgf')
    plt.savefig(output_path + '/train-val_acc.pdf')
    plt.close()


def save_set_stats(output_path, train_loader, unbalanced_train_loader, val_loader, test_loader, train_dataset,
                   seg=False):
    Set_analyst(given_set=train_dataset).bar_plot("train_set", output_path)
    if not seg:
        Set_analyst([train_loader, unbalanced_train_loader]).bar_plot("train", output_path)
        Set_analyst([val_loader]).bar_plot("val", output_path)
        Set_analyst([test_loader]).bar_plot("test", output_path)
