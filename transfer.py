import os
from experiments import Experimenter
import torch
from learning.models import PN2Net,DGCNNNet, UNet



class Transfer(Experimenter):
    def __init__(self, path, name, type):
        self.dataset_path = path
        self.dataset_name = name
        self.dataset_type = type

    def run(self, model_list, output_path, n_epochs, batch_size, learning_rate):


        for model in model_list:

            checkpoint = torch.load(model)
            model_name = checkpoint['model']
            print(model_name)

            if model_name == 'PN2Net':
                model_name = PN2Net
            if model_name == 'DGCNNNet':
                model_name = DGCNNNet
            if model_name == 'UNet':
                model_name = UNet

            test_acc, epoch_losses, train_accuracies, val_accuracies = self.subrun(output_path, model_name, n_epochs, batch_size, learning_rate, pretrained=checkpoint)


        # Load new dataset







if __name__ == '__main__':

    # Name of Dataset, careful the string matters!

    dataset_name = 'BIM_PC_T1'  # BIM_PC_T2, 'BIM_PC_T1', ModelNet10, ModelNet40
    print("retraining on {}".format(dataset_name))
    config = None
    ex = Experimenter(config, dataset_name)

    # fills list with all models to perform inference on - custom models
    model_list = ['../out/0/model_state_best_val.pth.tar'] # ['../out/0/model_state_best_val.pth.tar']
    output_path = '../out/0/trans/' + str()
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    trans = Transfer(ex.dataset_path, ex.dataset_name, ex.dataset_type)
    trans.run(model_list, output_path, n_epochs=50, batch_size=15, learning_rate=0.001)

    #print('inference for {} models with test {}'.format(i))