import os.path as osp
from sklearn.metrics import accuracy_score
import torch

class Results():
    def __init__(self):
        self.train_losses = []
        self.val_accuracy = []
        #self.val_aps = []
        self.val_rocs = []
        #self.test_aps = []
        #self.test_rocs = []
        self.correct = 0

    def update(self, train_loss=None, val_accuracy=None, val_roc=None):
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_accuracy is not None:
            self.val_accuracy


    def updateValAccuracy(self, label, pred, nb):
        acc = 0
        self.correct += pred.eq(label).sum().item()
        self.val_accuracy.append(self.correct/nb)

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
