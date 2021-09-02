
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





sns.set_style("white")
sns.set(rc={'figure.figsize':(11.7,8.27),'axes.facecolor':'white', 'figure.facecolor':'white'})

tsne = TSNE()

method = "KNN"
#method = "Mesh"

if method == "KNN":

    file1 = os.getcwd()
    file1 = os.path.join(file1, "resources/knn_rot2.csv")
    file2 = os.getcwd()
    file2 = os.path.join(file2, "resources/knn_rot3.csv")
   
    reader1 = csv.reader(open(file1, "rt"), delimiter=",")
    reader2 = csv.reader(open(file2, "rt"), delimiter=",")
    

    
    #mode = "allwithrotationvariant"
    mode = "twowithrotationvariant"
    #mode = "allwithrotationinvariant"
    #mode = "twowithrotationinvariant"


    if mode == "allwithrotationvariant":

        
        palette = sns.color_palette("tab20", 13)

        features = []
        labels = []

        for row in reader1:
            labels.append(row[0])
            row = row[1:]
            features.append(row)


        X_embedded = tsne.fit_transform(features)

        sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels,legend='full', palette = palette).set(title='kNN: Rotation Variant')


        #plt.scatter(X_embedded[:,0],y=X_embedded[:,1])

        plt.show()

    if mode == "allwithrotationinvariant":

        palette = sns.color_palette("tab20", 13)

        features = []
        labels = []

        for row in reader2:
            labels.append(row[0])
            row = row[1:]
            features.append(row)


        X_embedded = tsne.fit_transform(features)

        sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels,legend='full', palette = palette).set(title='kNN: Rotation Invariant')


        #plt.scatter(X_embedded[:,0],y=X_embedded[:,1])

        plt.show()
    if mode == "twowithrotationvariant":

        
        
        
        



        palette = sns.color_palette("bright", 3)

        features =[]
        labels =[]
        for row in reader1:
            if row[0] == "IfcFurnishingElement" or row[0] == "IfcFlowSegment" or row[0] == "IfcFlowTerminal":
                labels.append(row[0])
                row = row[1:]
                features.append(row)

        X_embedded = tsne.fit_transform(features)

        fig, ax = plt.subplots()
        x = X_embedded[:,0]
        y = X_embedded[:,1]
        sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels,legend='full', palette = palette)

        #plt.scatter(X_embedded[:,0],y=X_embedded[:,1])
        lnx = plt.plot([60,60], [0,1.5], color='black', linewidth=0.3)
        lny = plt.plot([0,100], [1.5,1.5], color='black', linewidth=0.3)
        lnx[0].set_linestyle('None')
        lny[0].set_linestyle('None')
        plt.xlim(x.min()*0.95, x.max()*1.05)
        plt.ylim(y.min()*0.95, y.max()*1.05)
        
        def hover(event):
            
            lnx[0].set_data([event.xdata, event.xdata], [0, 1.5])
            lnx[0].set_linestyle('--')
            lny[0].set_data([0,100], [event.ydata, event.ydata])
            lny[0].set_linestyle('--')
        
        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.title('kNN: Rotation Variant')

        plt.show()

    if mode == "twowithrotationinvariant":
        palette = sns.color_palette("bright", 3)

        features =[]
        labels =[]
        for row in reader2:
            if row[0] == "IfcFurnishingElement" or row[0] == "IfcFlowSegment" or row[0] == "IfcFlowTerminal":
                labels.append(row[0])
                row = row[1:]
                features.append(row)

        X_embedded = tsne.fit_transform(features)

        sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels,legend='full', palette = palette).set(title='kNN: Rotation Invariant')


        #plt.scatter(X_embedded[:,0],y=X_embedded[:,1])

        plt.show()

if method == "Mesh":

    file1 = os.getcwd()
    file1 = os.path.join(file1, "resources/mesh_rot2.csv")
    file2 = os.getcwd()
    file2 = os.path.join(file2, "resources/mesh_rot3.csv")
    
    reader1 = csv.reader(open(file1, "rt"), delimiter=",")
    reader2 = csv.reader(open(file2, "rt"), delimiter=",")

    method = "KNN"
    method = "Mesh"
    #mode = "allwithrotationvariant"
    #mode = "twowithrotationvariant"
    #mode = "allwithrotationinvariant"
    #mode = "twowithrotationinvariant"


    if mode == "allwithrotationvariant":

        palette = sns.color_palette("tab20", 13)

        features = []
        labels = []

        for row in reader1:
            labels.append(row[0])
            row = row[1:]
            features.append(row)


        X_embedded = tsne.fit_transform(features)

        sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels,legend='full', palette = palette).set(title='Mesh: Rotation Variant')


        #plt.scatter(X_embedded[:,0],y=X_embedded[:,1])

        plt.show()

    if mode == "allwithrotationinvariant":

        palette = sns.color_palette("tab20", 13)

        features = []
        labels = []

        for row in reader2:
            labels.append(row[0])
            row = row[1:]
            features.append(row)


        X_embedded = tsne.fit_transform(features)

        sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels,legend='full', palette = palette).set(title='Mesh: Rotation Invariant')


        #plt.scatter(X_embedded[:,0],y=X_embedded[:,1])

        plt.show()
    if mode == "twowithrotationvariant":
        palette = sns.color_palette("bright", 2)

        features =[]
        labels =[]
        for row in reader1:
            if row[0] == "IfcSlab" or row[0] == "IfcDoor":
                labels.append(row[0])
                row = row[1:]
                features.append(row)

        X_embedded = tsne.fit_transform(features)

        sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels,legend='full', palette = palette).set(title='Mesh: Rotation Variant')

        #plt.scatter(X_embedded[:,0],y=X_embedded[:,1])

        fig, ax = plt.subplots()
        #ax.scatter(X_embedded[:,0], X_embedded[:,1], c=labels.map(palette))
        plt.show


    if mode == "twowithrotationinvariant":
        palette = sns.color_palette("bright", 2)

        features =[]
        labels =[]
        for row in reader2:
            if row[0] == "IfcSlab" or row[0] == "IfcDoor":
                labels.append(row[0])
                row = row[1:]
                features.append(row)

        X_embedded = tsne.fit_transform(features)

        sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels,legend='full', palette = palette).set(title='Mesh: Rotation Invariant')


        #plt.scatter(X_embedded[:,0],y=X_embedded[:,1])

        plt.show()

    

