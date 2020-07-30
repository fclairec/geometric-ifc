__author__ = 'fiona.collins'

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)

# from tensorflow.keras.metrics import Accuracy
from tqdm import tqdm

from torch_geometric.data import DataLoader

import os.path as osp
import matplotlib.pyplot as plt
from numpy import concatenate as concat
from torch_geometric.utils import intersection_and_union as i_and_u

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
printout = 15


def pointnetloss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)


class Trainer:
    def __init__(self, model, output_path, max_patience=15):
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
            train_accuracies_batch = []
            for i, data in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = F.nll_loss(outputs, data.y)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    train_losses.append(loss.data)
                    pred = outputs.max(1)[1]
                    acc=pred.eq(data.y).sum().item() / train_loader.batch_size
                    train_accuracies_batch.append(acc)

                # for batch progress tracking in terminal
                if (i + 1) % printout == 0:
                    print('\nBatches {}-{}/{} (BS = {}) with loss {} and accuracy {}'.format(i - printout + 1, i,
                                                                                             len(train_loader),
                                                                                             train_loader.batch_size,
                                                                                             loss,
                                                                                             train_accuracies_batch[-1]))

            epoch_losses.append(torch.mean(torch.stack(train_losses, dim=0)))
            train_accuracies.append(np.mean(train_accuracies_batch))

            print("train correct")
            print(train_accuracies[-1])
            #train_acc = self.eval(train_loader, seg=False)
            print("eval correct")
            val_acc = self.eval(val_loader, seg=False)
            val_accuracies.append(val_acc)

            print("\nEpoch {} - train loss: {}, train acc: {} val acc: {}".format(epoch, epoch_losses[-1].item(),
                                                                                  train_accuracies[-1],
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
                                     'optimizer': optimizer.state_dict(),
                                     'num_output_classes': train_loader.dataset.num_classes
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

    def eval(self, data_loader, seg):
        self.model.eval()

        correct = 0
        for data in data_loader:
            data = data.to(device)
            # loss = F.nll_loss(self.model(data)[0], data.y)
            with torch.no_grad():
                pred = self.model(data).max(1)[1]
            correct += pred.eq(data.y).sum().item()

        if seg:
            acc = correct / len(data_loader.dataset.data.pos)
            print("correct {} / {}".format(correct, len(data_loader.dataset.data.pos)))
        else:
            acc = correct / len(data_loader.dataset)
            print("correct {} / {}".format(correct, len(data_loader.dataset)))

        return acc

    def test(self, data_loader, seg):
        self.model.eval()
        y_pred = []
        y_real = []
        correct = 0
        prob = []
        crit_points_list = []
        # ious = [[] for _ in range(data_loader.dataset.num_classes)]
        for data in data_loader:
            data = data.to(device)
            # loss = F.nll_loss(self.model(data)[0], data.y)
            with torch.no_grad():
                outputs = self.model(data)
                loss = F.nll_loss(outputs, data.y)
                SM = torch.max(torch.exp(outputs))
                pred = outputs.max(1)[1]
            correct += pred.eq(data.y).sum().item()

            y_pred = concat((y_pred, pred.cpu().numpy()))
            y_real = concat((y_real, data.y.cpu().numpy()))
            # crit_points_list.append(crit_points.cpu().numpy())
            prob.append(SM.cpu().numpy())

        if seg:
            acc = correct / len(data_loader.dataset.data.pos)
        else:
            acc = correct / len(data_loader.dataset)
        test_ious = self.calculate_sem_IoU(y_pred, y_real, data_loader.dataset.num_classes)

        return acc, y_pred, y_real, test_ious, prob

    def save_checkpoint(self, path, state, is_best):
        filename = 'model_state.pth.tar'
        path_out = osp.join(path, filename)
        torch.save(state, path_out)
        if is_best:
            filename = 'model_state_best_val.pth.tar'
            path_out = osp.join(path, filename)
            torch.save(state, path_out)

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


class Trainer_seg(Trainer):

    def train(self, train_loader, val_loader, num_epochs, optimizer):
        seg = True
        best_performance = 0

        epoch_losses = []
        train_accuracies = []
        val_accuracies = []
        """train_true_labels = []
        train_pred_labels = []"""

        for epoch in tqdm(range(num_epochs), 'processing epochs...'):
            self.model.train()
            train_losses = []
            correct = 0

            for i, data in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = F.nll_loss(outputs, data.y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.data * train_loader.batch_size)
                with torch.no_grad():
                    pred = outputs.max(1)[1]
                correct += pred.eq(data.y).sum().item()
                pred_np = pred.detach().cpu().numpy()

                """train_true_labels.append(data.y.detach().cpu().numpy())
                train_pred_labels.append(pred_np)"""

                # TODO: careful! This needs to be changed when switching from shape to node classification

                # for batch progress tracking in terminal
                if (i + 1) % printout == 0:
                    print('\nBatches {}-{}/{} (BS = {}) with loss {}'.format(i - printout + 1, i, len(train_loader),
                                                                             train_loader.batch_size, train_losses[-1]))

            train_acc = correct / len(train_loader.dataset.data.pos)

            epoch_losses.append(torch.mean(torch.stack(train_losses, dim=0)).item())

            print("train correct")
            # train_acc = self.eval(train_loader)
            print("eval correct")
            val_acc = self.eval(val_loader)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            print("\nEpoch {} - train loss: {}, train acc: {} val acc: {}".format(epoch, epoch_losses[-1],
                                                                                  train_acc,
                                                                                  val_acc
                                                                                  ))

            is_best = val_acc > best_performance
            best_performance = max(val_acc, best_performance)
            # saves state dictionary of every epoch (overwrites), additionally saves best
            self.save_checkpoint(self.output_path,
                                 {
                                     'model': type(self.model).__name__,
                                     'epoch': epoch + 1,
                                     'state_dict': self.model.state_dict(),
                                     'best_train_acc': best_performance,
                                     'optimizer': optimizer.state_dict(),
                                     'num_output_classes': train_loader.dataset.num_classes
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

        """# fill for IoU
        train_true_labels = np.concatenate(train_true_labels)
        train_pred_labels = np.concatenate(train_pred_labels)
        # train_ious = self.calculate_sem_IoU(train_pred_labels, train_true_labels)"""

        return epoch_losses, train_accuracies, val_accuracies
