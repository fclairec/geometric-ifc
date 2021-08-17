from helpers.set_plot import Set_analyst
from torch_geometric.utils import degree
from helpers.visualize import vis_graph
import os


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
    save_plot = False
    if save_plot:
        for i, data in enumerate(test_loader):
            #if i ==20: break
            idx, x = data.edge_index[1], data.x
            deg = degree(idx, data.num_nodes, dtype=torch.long)
            _, counts = train_dataset.data.y.unique(return_counts=True)
            title = " , max node degrees " + str(deg.max().item()) + " , max edge length " + str(np.round(data.edge_attr.max().item(),2))

            #data=train_dataset.__getitem__(1)
            vis_graph(data, out_path=output_path, classmap=train_dataset.classmap, title = title, i=i)
            a=0


    print("printing set stats")
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