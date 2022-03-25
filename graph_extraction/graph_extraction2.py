from unicodedata import category
import networkx as nx
import matplotlib.pyplot as plt
import csv
import pandas as pd
from pandas import DataFrame
import math
import argparse
import os
import pygraphviz
from itertools import count


## argument for csv file
parser = argparse.ArgumentParser(description='Make network graphs')
parser.add_argument('filename', default='/home/asalman/ifcwork/graph_extraction',
                    type=str, help='csv file to make graph')
args = parser.parse_args()
file = args.filename



file = os.path.join("/home/asalman/ifcwork/graph_extraction", file)
df = pd.read_csv(file, keep_default_na=False)
Data = open(file, 'r')
csv_reader = csv.reader(Data)



## converting pandas to each category lists
StartElements = df['Start node'].to_list()
EndElements = df['End node'].to_list()
StartCategories = df['Start Category'].to_list()

pairs = []
labels = []
attrdir = {}

for i in range(len(StartElements)):
    pairs.append((StartElements[i], EndElements[i]))
    labels.append(StartCategories[i])
    attr = { 'category': StartCategories[i]}
    attrdir[StartElements[i]] = attr


G = nx.Graph()

G.add_nodes_from(StartElements)
G.add_edges_from(pairs)

pos = nx.nx_agraph.graphviz_layout(G) ## this takes the positions of nodes

#pos = nx.nx_agraph.graphviz_layout(G) ## this takes the positions of nodes


nx.set_node_attributes(G, attrdir)

print(attrdir)

groups = set(nx.get_node_attributes(G,'category').values())
mapping = dict(zip(sorted(groups),count()))

colors = []

for n in G.nodes():
    if bool(G.nodes[n]):

        colors.append(mapping[G.nodes[n]['category']])
            

    else:
        colors.append('5')
            #
        print(n)
        


fig, ax = plt.subplots()
nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.3)
nodes = nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=colors, cmap=plt.cm.jet)



annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

idx_to_node_dict = {}
for idx, node in enumerate(G.nodes):
    idx_to_node_dict[idx] = node

def update_annot(ind):
    node_idx = ind["ind"][0]
    node = idx_to_node_dict[node_idx]
    xy = pos[node]
    annot.xy = xy
    node_attr = {'node': node}
    node_attr.update(G.nodes[node])
    text = '\n'.join(f'{k}: {v}' for k, v in node_attr.items())
    annot.set_text(text)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = nodes.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

#nx.draw_networkx(G)
plt.draw()
plt.show()

