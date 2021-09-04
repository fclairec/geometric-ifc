
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
#label = 'Predicted'

mode = "Rotation Variant"
#mode = "Rotation Invariant"

numOfClasses  = 13
#numOfClasses = 2
#numOfClasses = 3

subclasses = ['IfcFurnishingElement', 'IfcFlowSegment']

def scatterplot(directory, numOfClasses, label):
    

    dir = directory
    featureSpaceFile = os.path.join(dir, 'featurespace.csv')
    predictionsFile = os.path.join(dir, 'trueVspred-withId.csv')

    df = pd.read_csv(predictionsFile)
    dict_map = {1: 'IfcFlowController', 2: 'IfcFlowFitting',3: 'IfcFlowSegment',
        4: 'IfcFlowTerminal',5: 'IfcColumn',6: 'IfcFurnishingElement',7: 'IfcStair',8: 'IfcDoor',
        9: 'IfcSlab',10: 'IfcWall',11: 'IfcWindow',12: 'IfcRailing',0: 'IfcDistributionControlElement'}
    df.y_pred = [dict_map[item] for item in df.y_pred]
    df.y_real = [dict_map[item] for item in df.y_real]
   
    print(featureSpaceFile)

    featureSpace = csv.reader(open(featureSpaceFile, "rt"), delimiter=",")
   
    
    features = []
    labels = []
    label = label
    print(label)
            
    next(featureSpace)
    

    if numOfClasses == 13:

        for row in featureSpace:
            
            row = row[1:]
            features.append(row)
                 
            
        if label == 'Real':
            for row in df.y_real:
                
                labels.append(row)
                
        else:
            for row in df.y_pred:
                labels.append(row)

    if numOfClasses == 2:

        for row in featureSpace:
            if row[0] == subclasses[0] or row[0] == subclasses[1]:
                labels.append(row[0])
                row = row[1:]
                features.append(row)

    if numOfClasses == 3:
        
        while len(subclasses) == 3:
            for row in featureSpace:
                if row[0] == subclasses[0] or row[0] == subclasses[1] or row[0] == subclasses[2]:
                    labels.append(row[0])
                    row = row[1:]
                    features.append(row)
        else:
            print('Error: Enter at least 3 subclasses')
            exit()
            
        
        


    tsne = TSNE()
    X_embedded = tsne.fit_transform(features)
    x = X_embedded[:,0]
    y = X_embedded[:,1]

    fig, ax = plt.subplots()
    labels, index = np.unique(labels, return_inverse=True)
    
    sc = ax.scatter(x, y, marker = '.', c = index, alpha = 1)
    ax.legend(sc.legend_elements()[0], labels)

    plt.xlim(x.min()*0.95, x.max()*1.05)
    plt.ylim(y.min()*0.95, y.max()*1.05)
            
    annot = ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points")
    annot.set_visible(False)

    def hover(event):

        # check if event was in the axis
        if event.inaxes == ax:
            
            cont, ind = sc.contains(event)
            if cont:
                # change annotation position
                annot.xy = (event.xdata, event.ydata)
                # write the name of every point contained in the event
                annot.set_text("{}".format(', '.join([str(n) for n in ind["ind"]])))
                annot.set_visible(True)    
            else:
                annot.set_visible(False)
        

        plt.draw()
            
                      
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.title(method + ': ' + mode)
    plt.show()  




tsne = TSNE()



if method == "KNN":

    current_dir = os.getcwd()
    file1 = os.path.join(current_dir, "resources/knn_easy")
    file2 = os.path.join(current_dir, "resources/knn_full")

else:

    current_dir = os.getcwd()
    file1 = os.path.join(current_dir, "resources/mesh_easy")
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
        


        

