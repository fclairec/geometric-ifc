
from matplotlib.pyplot import legend
import numpy as np
from seaborn import palettes
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os



## Chose the config below 


method = "KNN"
#method = "Mesh"

label = 'Real'
#label_pred = 'Predicted'

mode = "Rotation Variant"
#mode = "Rotation Invariant"

numOfClasses  = 13
#numOfClasses = 2
#numOfClasses = 3
#numOfClasses = 4


subclasses = ['IfcFurnishingElement', 'IfcFlowSegment']
subclasses = ['IfcFurnishingElement', 'IfcFlowTerminal', 'IfcFlowController']

def scatterplot(directory, numOfClasses, label):
    

    dir = directory
    #featureSpaceFile = os.path.join(dir, 'featurespace.csv')
    finalembeddings = os.path.join(dir, 'finalembeddings.csv')
    predictionsFile = os.path.join(dir, 'trueVspred-withId.csv')

    df = pd.read_csv(predictionsFile)
    dict_map = {1: 'IfcFlowController', 2: 'IfcFlowFitting',3: 'IfcFlowSegment',
        4: 'IfcFlowTerminal',5: 'IfcColumn',6: 'IfcFurnishingElement',7: 'IfcStair',8: 'IfcDoor',
        9: 'IfcSlab',10: 'IfcWall',11: 'IfcWindow',12: 'IfcRailing',0: 'IfcDistributionControlElement'}
    df.y_pred = [dict_map[item] for item in df.y_pred]
    df.y_real = [dict_map[item] for item in df.y_real]
    df.id = [item.split("/")[-1] for item in df.object_id]
   
    print(finalembeddings)

    finalembeddings = csv.reader(open(finalembeddings, "rt"), delimiter=",")
   
    
    features = []
    labels = []
    label = label
    ids = []
    pred_labels = []
    print(label)
            
    next(finalembeddings)
    

    if numOfClasses == 13:

        for row in finalembeddings:
            class_name = row[0].split("/")[-1].split("_")[-1].split(".")[0]
            labels.append(class_name)
            ids.append(row[0].split("/")[-1])
            pred = df.loc[df['object_id'] == row[0]].y_pred.values[0]
            row = row[1:]
            features.append(row)
            pred_labels.append(pred)




    for row in finalembeddings:
        class_name = row[0].split("/")[-1].split("_")[-1].split(".")[0]
        if class_name in subclasses:
            labels.append(class_name)
            ids.append(row[0].split("/")[-1])
            pred = df.loc[df['object_id'] == row[0]].y_pred.values[0]
            row = row[1:]
            features.append(row)
            pred_labels.append(pred)

        else:
            continue




    tsne = TSNE()
    X_embedded = tsne.fit_transform(features)
    x = X_embedded[:,0]
    y = X_embedded[:,1]

    fig, ax = plt.subplots()
    labels, index = np.unique(labels, return_inverse=True)


    sc = ax.scatter(x, y, marker = '.', c = index, alpha = 1, cmap='tab20')
    ax.legend(sc.legend_elements()[0], labels,loc=2, prop={'size': 6})

    plt.xlim(x.min()*0.95, x.max()*1.05)
    plt.ylim(y.min()*0.95, y.max()*1.05)
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.axis('off')

    annot = ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points")
    annot.set_visible(True)

    def hover(event):

        # check if event was in the axis
        if event.inaxes == ax:
            
            cont, ind = sc.contains(event)
            print(ind)
            if cont:
                # change annotation position
                annot.xy = (event.xdata, event.ydata)
                # write the name of every point contained in the event
                annot.set_text("{}".format(', '.join(["true:"+str(ids[n])+" and pred:"+str(pred_labels[n]) for n in ind["ind"]])))
                annot.set_visible(True)    
            else:
                annot.set_visible(False)
        

        plt.draw()
            
                      
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()




tsne = TSNE()



if method == "KNN":

    current_dir = os.getcwd()
    file1 = os.path.join(current_dir, "../../data/05_kann/4_clas/inference")
    file2 = os.path.join(current_dir, "resources/knn_full")

else:

    current_dir = os.getcwd()
    file1 = os.path.join(current_dir, "../../data/3_clas/inference")
    file2 = os.path.join(current_dir, "resources/mesh_full")
   
   

if mode == "Rotation Variant":
    if label == 'Prediction' and numOfClasses != 13: 
        print('Error: Prediction cofiguration can only be used with all classes')

    else:
        scatterplot(file1,numOfClasses,label)
        

    
    
if mode == "Rotation Invariant":

    if label == 'Prediction' and numOfClasses != 13: 
        print('Error: Prediction cofiguration can only be used with all classes')

    else:
        scatterplot(file2,numOfClasses,label)
        


        
