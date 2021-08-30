from matplotlib.pyplot import legend
import numpy as np
from seaborn import palettes
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd


sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("tab20", 13)

file = '/home/asalman/ifcwork/ifcworkspi2021/geometric-ifc/resources/feature_space.csv'
file2 = '/home/asalman/ifcwork/ifcworkspi2021/geometric-ifc/resources/feature_space_withlabels.csv'
reader = csv.reader(open(file, "rt"), delimiter=",")
reader2 = csv.reader(open(file2, "rt"), delimiter=",")
features = np.array(list(reader)).astype("float")
labels = np.array(list(reader2)).astype("str")


def Extract(label):
    return [item[0] for item in label]
      
# Driver code

labels = Extract(labels)




X, y = load_digits(return_X_y=True)


tsne = TSNE()

X_embedded = tsne.fit_transform(features)

sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels,legend='full', palette = palette)

#plt.scatter(X_embedded[:,0],y=X_embedded[:,1])

plt.show()

