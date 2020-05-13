__author__ = 'fiona.collins'

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from ignite.metrics import Accuracy
from sklearn.metrics import accuracy_score

#from tensorflow.keras.metrics import Accuracy
from tqdm import tqdm

from torch_geometric.data import DataLoader

import os.path as osp
import matplotlib.pyplot as plt
from numpy import concatenate as concat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
printout = 15


class Trainer:
    def __init__(self, model, output_path, max_patience=2):
        self.model = model
        self.output_path = output_path
        self.max_patience = max_patience
        self.patience = self.max_patience

    def train(self, train_loader, val_loader, num_epochs, optimizer):
        best_performance = 0

        epoch_losses = []
        train_accuracies = []
        val_accuracies = []
        for epoch in tqdm(range(num_epochs), 'processing epochs...'):
            self.model.train()
            train_losses = []
            for i, data in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                loss = F.nll_loss(self.model(data), data.y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.data)

                # for batch progress tracking in terminal
                if (i + 1) % printout == 0:
                    print('\nBatches {}-{}/{} (BS = {})'.format(i - printout + 1, i, len(train_loader),
                                                                train_loader.batch_size))

            epoch_losses.append(torch.mean(torch.stack(train_losses, dim=0)))

            print("train correct")
            train_acc = self.eval(train_loader)
            print("eval correct")
            val_acc = self.eval(val_loader)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            print("\nEpoch {} - train loss: {}, train acc: {} val acc: {}".format(epoch, epoch_losses[-1].item(),
                                                                                  train_acc,
                                                                                  val_acc))

            is_best = val_acc > best_performance
            best_performance = max(val_acc, best_performance)
            # saves state dictionary of every epoch (overwrites), additionally saves best
            self.save_checkpoint(self.output_path,
                                 {
                                     'model': type(self.model).__name__,
                                     'epoch': epoch + 1,
                                     'state_dict': self.model.state_dict(),
                                     'best_val_acc': best_performance,
                                     'optimizer': optimizer.state_dict()
                                 }, is_best
                                 )

            # if we just found a better model (better validation accuracy)
            # then the window for early stop is set back to the max patience
            if is_best:
                self.patience = self.max_patience
            else:
                self.patience -= 1

            if self.patience == 0:
                # early stopping
                # load best model back into memory
                model_load = self.output_path + '/model_state_best_val.pth.tar'
                checkpoint = torch.load(model_load)
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                self.model.to(device)
                return epoch_losses, train_accuracies, val_accuracies



        return epoch_losses, train_accuracies, val_accuracies

    def eval(self, data_loader):
        self.model.eval()

        correct=0
        for data in data_loader:
            data = data.to(device)
            loss = F.nll_loss(self.model(data), data.y)
            with torch.no_grad():
                pred = self.model(data).max(1)[1]
            correct += pred.eq(data.y).sum().item()

        acc = correct / len(data_loader.dataset)
        print ("correct {} / {}". format(correct, len(data_loader.dataset)))

        return acc

    def test(self, data_loader):
        self.model.eval()
        y_pred=[]
        y_real=[]
        correct=0
        for data in data_loader:
            data = data.to(device)
            loss = F.nll_loss(self.model(data), data.y)
            with torch.no_grad():
                pred = self.model(data).max(1)[1]
                m = self.model(data)
                d= m.min(1)
                a=d[1]
            correct += pred.eq(data.y).sum().item()
            y_pred = concat((y_pred, pred.cpu().numpy()))
            y_real = concat((y_real, data.y.cpu().numpy()))

        acc = correct / len(data_loader.dataset)

        return acc, y_pred, y_real

    def save_checkpoint(self, path, state, is_best):
        filename = 'model_state.pth.tar'
        path_out = osp.join(path, filename)
        torch.save(state, path_out)
        if is_best:
            filename = 'model_state_best_val.pth.tar'
            path_out = osp.join(path, filename)
            torch.save(state, path_out)
