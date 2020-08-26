import os.path as osp
from sklearn.metrics import accuracy_score
import torch
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib as mpl
import numpy as np
import os
from helpers.set_plot import Set_analyst





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
    return num_trainable_params


def calculate_sem_IoU(pred_np, seg_np, num_classes):
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
                      val_accuracies, WRITE_DF_TO_, plot_name, seg=False):
    plt = mpl.pyplot
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    params = {'text.usetex': True,
              'font.size': 11,
              'font.family': 'lmodern'
              }
    plt.rcParams.update(params)


    real_target_names = [test_dataset.classmap[i] for i in np.unique(np.array(test_dataset.data.y))]
    plot_name.replace('_', ' ')
    print(plot_name)

    if seg:
        test_ious = calculate_sem_IoU(y_pred, y_real, test_dataset.num_classes)

        # write Test IOUS
        df0 = DataFrame(test_ious, index=real_target_names, columns=[plot_name])
        file_ious_csv = output_path + '/test_ious.csv'
        df0.to_csv(file_ious_csv)
        if 'to_latex' in WRITE_DF_TO_:
            file_ious_tex = output_path + '/test_ious' + plot_name + '.tex'
            df0.to_latex(file_ious_tex, caption=plot_name, index=False)


    if not seg:
        # Confusion matrix
        conf_mat = confusion_matrix(y_true=y_real, y_pred=y_pred, normalize='true')

        df1 = DataFrame(conf_mat, index=real_target_names, columns=real_target_names)

        file_confmat_csv = output_path + '/confmat.csv'
        df1.to_csv(file_confmat_csv)
        if 'to_latex' in WRITE_DF_TO_:
            file_confmat_tex = output_path + '/confmat' + plot_name + '.tex'
            df1.to_latex(file_confmat_tex, caption=plot_name)
            print("latex written")

        # Classification report
        class_rep = classification_report(y_true=y_real, y_pred=y_pred, target_names=real_target_names,
                                          output_dict=True)
        df2 = DataFrame(class_rep).transpose()
        file_classrep_csv = output_path + '/class_report.csv'
        df2.to_csv(file_classrep_csv)
        if 'to_latex' in WRITE_DF_TO_:
            file_classrep_tex = output_path + '/class_report' + plot_name + '.tex'
            df2.to_latex(file_classrep_tex, caption=plot_name)

    plot_name.replace('_', ' ')
    # plot epoch losses
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(range(len(epoch_losses)), epoch_losses, label='training loss', color='steelblue')
    ax.legend()
    ax.set_title("Train loss" + plot_name, fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    ax.set_xticks(range(len(epoch_losses)))
    ax.set_xticklabels(range(len(epoch_losses)), fontsize=12)
    if 'to_latex' in WRITE_DF_TO_:
        plt.savefig(output_path + '/train_loss' + plot_name + '.pgf')
        import tikzplotlib
        tikzplotlib.save("test.tex")

    #plt.savefig(output_path + '/train_loss.pdf')
    plt.close()
    plt.clf()


    # plot train and val accuracies
    fig, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(range(len(train_accuracies)), train_accuracies, label='Training accuracies', color='steelblue')
    ax2.plot(range(len(train_accuracies)), val_accuracies, label='Validation accuracies', color='indianred')
    ax2.legend(fontsize=16)
    ax2.set_title("Train and validation accuracies" + plot_name, fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy score', fontsize=16)
    ax2.set_xticks(range(len(train_accuracies)))
    ax2.set_xticklabels(range(len(train_accuracies)), fontsize=12)
    plot_name.replace('_', ' ')
    if 'to_latex' in WRITE_DF_TO_:
        plt.savefig(output_path + '/train-val_acc' + plot_name + '.pgf')
    #plt.savefig(output_path + '/train-val_acc.pdf')
    plt.close()
    plt.clf()

    # save epoch losses, train_loss, train_accuracies, validation accuracies
    dic = {'train loss': epoch_losses, 'train accuracy': train_accuracies, 'validation accuracy': val_accuracies}
    df4 = DataFrame(dic)
    file_epochresults_csv = output_path + '/epoch_results.csv'
    df4.to_csv(file_epochresults_csv)
    plot_name.replace('_',' ')
    if 'to_latex' in WRITE_DF_TO_:
        file_epochresults_txt = output_path + '/epoch_results' + plot_name + '.tex'
        df4.to_latex(file_epochresults_txt, caption=plot_name, index=False)

    plt.close('all')

def print_set_csv(l_trainset, l_testset,val_dataset, output_path, classmap, plotname):

    filename = 'results_dataset_' + plotname + '.csv'
    path = os.path.join(output_path, filename)
    total_trainset= l_trainset[0]+val_dataset[0]
    with open(path, encoding='utf-8-sig', mode='w') as fp:
        fp.write('Train dataset\n')
        for tag, count in total_trainset.items():
            tag = classmap[tag]
            fp.write('{},{}\n'.format(tag, count))
        fp.write('\nTest dataset\n')
        for tag, count in l_testset[0].items():
            tag = classmap[tag]
            fp.write('{},{}\n'.format(tag, count))

    return


def save_set_stats(output_path, train_loader, test_loader, train_dataset, test_dataset, val_dataset, unbalanced_train_loader=None, val_loader=None, seg=False):
    Set_analyst(given_set=train_dataset).bar_plot("train_set", output_path)
    l_trainset = Set_analyst(given_set=train_dataset).class_counter()
    l_testset = Set_analyst(given_set=test_dataset).class_counter()
    l_valset = Set_analyst(given_set=val_dataset).class_counter()
    print_set_csv(l_trainset, l_testset, l_valset, output_path, train_dataset.classmap, plotname='datasets')

    l_trainloader = Set_analyst([train_loader]).class_counter()
    l_testloader = Set_analyst([test_loader]).class_counter()
    l_valloader =  Set_analyst([val_loader]).class_counter()
    print_set_csv(l_trainloader, l_testloader, l_valloader, output_path, train_dataset.classmap, plotname='dataloaders')

    #if not seg:
        #Set_analyst([train_loader, unbalanced_train_loader]).bar_plot("train", output_path)
        #Set_analyst([val_loader]).bar_plot("val", output_path)
        #Set_analyst([test_loader]).bar_plot("test", output_path)
