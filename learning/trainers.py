__author__ = 'fiona.collins'

import numpy as np
import torch
import torch.nn.functional as F
import os
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
import time
import statistics

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
    def __init__(self, model, output_path, max_patience=80):
        self.model = model
        self.output_path = output_path
        self.max_patience = max_patience
        self.patience = self.max_patience

    def train(self, train_loader, val_loader, num_epochs, optimizer):
        best_performance = 0

        epoch_losses = []
        train_accuracies = []
        val_accuracies = []
        epoch_times = []

        for epoch in tqdm(range(num_epochs), 'processing epochs...'):
            t0 = time.time()
            self.model.train()
            train_losses = []
            train_accuracies_batch = []
            for i, data in enumerate(train_loader):
                #if i == 3: break
                data = data.to(device)
                optimizer.zero_grad()
                outputs, _ = self.model(data)
                loss = F.nll_loss(outputs, data.y)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    # train losses per batch
                    train_losses.append(loss.data*data.num_graphs)
                    pred = outputs.max(1)[1]
                    # train accuracies per batch
                    acc = pred.eq(data.y).sum().item() / data.num_graphs
                    train_accuracies_batch.append(acc)


                # for batch progress tracking in terminal
                if (i + 1) % printout == 0:
                    print('\nBatches {}-{}/{} (BS = {}) with loss {} and accuracy {}'.format(i - printout + 1, i,
                                                                                             len(train_loader),
                                                                                             train_loader.batch_size,
                                                                                             loss,
                                                                                             train_accuracies_batch[-1]))
            epoch_time = time.time() - t0
            epoch_times.append(epoch_time)


            epoch_losses.append(torch.mean(torch.stack(train_losses, dim=0)).item())
            train_accuracies.append(np.mean(train_accuracies_batch))

            print("train correct")
            print(train_accuracies[-1])
            #train_acc = self.eval(train_loader, seg=False)
            print("eval correct")
            val_acc = self.eval(val_loader, seg=False)
            val_accuracies.append(val_acc)

            print("\nEpoch {} - train loss: {}, train acc: {} val acc: {}".format(epoch, epoch_losses[-1],
                                                                                  train_accuracies[-1],
                                                                                  val_acc))

            is_best = val_acc > best_performance
            best_performance = max(val_acc, best_performance)
            # saves state dictionary of every epoch (overwrites), additionally saves best
            self.save_checkpoint(self.output_path,
                                 {
                                     'model': type(self.model).__name__,
                                     'epoch': epoch,
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
                print("reduce patience")

            if self.patience == 0 or epoch+1 == num_epochs:
                # early stopping
                # load best model back into memory
                model_load = self.output_path + '/model_state_best_val.pth.tar'
                checkpoint = torch.load(model_load)
                epoch_test = checkpoint['epoch']
                print("checkpoint from epoch {} loaded" .format(epoch_test))
                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
                optimizer.load_state_dict(checkpoint['optimizer'])
                self.model.to(device)
                mean_epoch_time = statistics.mean(epoch_times)
                return epoch_losses, train_accuracies, val_accuracies, epoch_test, mean_epoch_time
        return

    def eval(self, data_loader, seg):
        self.model.eval()

        correct = 0
        for i, data in enumerate(data_loader):
            #i == 3: break
            data = data.to(device)
            # loss = F.nll_loss(self.model(data)[0], data.y)
            with torch.no_grad():
                output, _ = self.model(data)
                pred = output.max(1)[1]
            correct += pred.eq(data.y).sum().item()

        if seg:
            acc = correct / len(data_loader.dataset.data.pos)
            print("val acc {}".format(acc))
        else:
            acc = correct / len(data_loader.dataset)
            print("val acc {}".format(acc))

        return acc

    def test(self, data_loader, seg, save_pred=False, prediction_path=None, inf_dataloader=None):
        self.model.eval()
        y_pred = []
        y_real = []
        correct = 0
        prob = []
        crit_points_list = []
        if save_pred and inf_dataloader is not None:
            print("running inference per batch with prediction printouts")
            if not os.path.exists(prediction_path):
                os.makedirs(prediction_path)

            for i, iterator in enumerate(zip(data_loader, inf_dataloader)):
                data, inf_data = iterator
                #i == 3: break
                data = data.to(device)
                # loss = F.nll_loss(self.model(data)[0], data.y)
                with torch.no_grad():
                    outputs,_ = self.model(data)
                    loss = F.nll_loss(outputs, data.y)
                    SM = torch.max(torch.exp(outputs))
                    pred = outputs.max(1)[1]

                    one = inf_data.pos
                    # Check if normalized indexes correspond to real ones
                    assert inf_data.y.numpy().all()==data.y.cpu().numpy().all()

                    nonnorm_labeled = np.concatenate((inf_data.pos.numpy(), data.y.cpu().numpy().reshape((-1, 1))), axis=1)
                    # add print of labels
                    np.savetxt(os.path.join(self.output_path, str(i) + ".txt"), nonnorm_labeled, delimiter=',')

                correct += pred.eq(data.y).sum().item()

                y_pred = concat((y_pred, pred.cpu().numpy()))
                y_real = concat((y_real, data.y.cpu().numpy()))
                # crit_points_list.append(crit_points.cpu().numpy())
                # prob.append(SM.cpu().numpy())

        else:
            # Loop for testing without writing results
            for i, data in enumerate(data_loader):
                #if i == 3: break
                data = data.to(device)
                # loss = F.nll_loss(self.model(data)[0], data.y)
                with torch.no_grad():
                    outputs, _ = self.model(data)
                    loss = F.nll_loss(outputs, data.y)
                    SM = torch.max(torch.exp(outputs))
                    pred = outputs.max(1)[1]

                correct += pred.eq(data.y).sum().item()

                y_pred = concat((y_pred, pred.cpu().numpy()))
                y_real = concat((y_real, data.y.cpu().numpy()))
                # crit_points_list.append(crit_points.cpu().numpy())
                #prob.append(SM.cpu().numpy())

        # Calculate Accuracies
        if seg:
            # average over the number of nodes
            acc = correct / len(data_loader.dataset.data.pos)
        else:
            # average over the number of graphs
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


class Trainer_seg(Trainer):

    def train(self, train_loader, val_loader, num_epochs, optimizer, best_performance=0):
        seg = True

        epoch_losses = []
        train_accuracies = []
        val_accuracies = []
        """train_true_labels = []
        train_pred_labels = []"""

        for epoch in tqdm(range(num_epochs), 'processing epochs...'):
            self.model.train()
            train_losses = []
            correct = 0
            train_accuracies_batch = []

            for i, data in enumerate(train_loader):
                #i == 3: break
                data = data.to(device)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = F.nll_loss(outputs, data.y)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    # Train losses per batch
                    train_losses.append(loss.data * len(data.pos))
                    pred = outputs.max(1)[1]
                    # Train accuracies per bach
                    acc = pred.eq(data.y).sum().item() / len(data.pos)
                    train_accuracies_batch.append(acc)
                #correct += pred.eq(data.y).sum().item()
                pred_np = pred.detach().cpu().numpy()


                # for batch progress tracking in terminal
                if (i + 1) % printout == 0:
                    print('\nBatches {}-{}/{} (BS = {}) with loss {} and accuracy {}'.format(i - printout + 1, i, len(train_loader),
                                                                             train_loader.batch_size, train_losses[-1], train_accuracies_batch[-1]))

            epoch_losses.append(torch.mean(torch.stack(train_losses, dim=0)).item())
            train_accuracies.append(np.mean(train_accuracies_batch))

            print("train correct")
            print(train_accuracies[-1])
            print("eval correct")
            val_acc = self.eval(val_loader, seg=True)
            val_accuracies.append(val_acc)

            print("\nEpoch {} - train loss: {}, train acc: {} val acc: {}".format(epoch, epoch_losses[-1],
                                                                                  train_accuracies[-1],
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
                print("reduce parience")

            if self.patience == 0 or epoch+1 == num_epochs:
                # early stopping
                # load best model back into memory
                model_load = self.output_path + '/model_state_best_val.pth.tar'
                checkpoint = torch.load(model_load)
                epoch_test = checkpoint['epoch']
                print("checkpoint from epoch {} loaded" .format(epoch_test))
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                self.model.to(device)
                return epoch_losses, train_accuracies, val_accuracies, epoch_test

        """# fill for IoU
        train_true_labels = np.concatenate(train_true_labels)
        train_pred_labels = np.concatenate(train_pred_labels)
        # train_ious = self.calculate_sem_IoU(train_pred_labels, train_true_labels)"""

        return
