import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.manifold import TSNE
import seaborn as sns

# file = '/home/asalman/ifcwork/ifcworkspi2021/geometric-ifc/resources/knn_rot2_new.csv'
# df = pd.read_csv(file)
# labels = df['Category'].values
# palette = sns.color_palette("tab20", 13)

# reader = csv.reader(open(file, "rt"), delimiter=",")

# features = []
        
# next(reader, None)

# for row in reader:
    
#     row = row[1:]
#     features.append(row)

# tsne = TSNE()

# X_embedded = tsne.fit_transform(features)
# fig, ax = plt.subplots()
# x = X_embedded[:,0]
# y = X_embedded[:,1]

file = '/home/asalman/ifcwork/ifcworkspi2021/geometric-ifc/resources/happiness.csv'
df = pd.read_csv(file)
x_name = 'Healthy life expectancy'
y_name = 'Freedom to make life choices'
tooltip_name = 'Country name'
x = df[x_name]
y = df[y_name]
tt = df[tooltip_name].values
df
fig, ax = plt.subplots(1, figsize=(12,6))
ax.scatter(x,y)
sc = ax.scatter(x,y)
plt.xlabel(x_name)
plt.ylabel(y_name)
#plt.scatter(X_embedded[:,0],y=X_embedded[:,1])
lnx = plt.plot([60,60], [0,1.5], color='black', linewidth=0.3)
lny = plt.plot([0,100], [1.5,1.5], color='black', linewidth=0.3)
lnx[0].set_linestyle('None')
lny[0].set_linestyle('None')
plt.xlim(x.min()*0.95, x.max()*1.05)
plt.ylim(y.min()*0.95, y.max()*1.05)
        
annot = ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points")
annot.set_visible(False)

def hover(event):

    # check if event was in the axis
    if event.inaxes == ax:
                # draw lines and make sure they're visible
        lnx[0].set_data([event.xdata, event.xdata], [0, 1.5])
        lnx[0].set_linestyle('--')
        lny[0].set_data([0,100], [event.ydata, event.ydata])
        lny[0].set_linestyle('--')
        lnx[0].set_visible(True)
        lny[0].set_visible(True)
                
                # get the points contained in the event
        cont, ind = sc.contains(event)
        if cont:
            # change annotation position
            annot.xy = (event.xdata, event.ydata)
            # write the name of every point contained in the event
            annot.set_text("{}".format(', '.join([tt[n] for n in ind["ind"]])))
            annot.set_visible(True)    
        else:
            annot.set_visible(False)
    else:
        lnx[0].set_visible(False)
        lny[0].set_visible(False)
        
        
    
            
fig.canvas.mpl_connect("motion_notify_event", hover)

plt.title('kNN: Rotation Variant')

plt.show()  