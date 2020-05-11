import os.path as osp
from sklearn.metrics import accuracy_score

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