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
elements = df['Entity'].to_list()
wall = df['Wall'].to_list()
stair = df['Stair'].to_list()
slab = df['Slab'].to_list()
furnishing = df['FurnishingElement'].to_list()
flowT = df['Flowterminal'].to_list()
flowS = df['FlowSegment'].to_list()
flowF = df['FlowFitting'].to_list()
flowX = df['FlowController'].to_list()
dist = df['DistributionControlElement'].to_list()
column = df['Column'].to_list()
window = df['Window'].to_list()
door = df['Door'].to_list()

pairs = []
attrdi = {}


# looping through each element to make edges. This can be improved in terms of implementation
for i in range(len(elements)):

    if wall[i] != '':
        pairs.append((elements[i], wall[i]))
        attr = {'category': 'wall'}
        attrdi[wall[i]] = attr

for i in range(len(elements)):

    if slab[i] != '':
        pairs.append((elements[i], slab[i]))
        attr = {'category': 'slab'}
        attrdi[slab[i]] = attr

for i in range(len(elements)):

    if door[i] != '':
        pairs.append((elements[i], door[i]))
        attr = {'category': 'door'}
        attrdi[door[i]] = attr

for i in range(len(elements)):

    if furnishing[i] != '':
        pairs.append((elements[i], furnishing[i]))
        attr = {'category': 'furnishing_element'}
        attrdi[furnishing[i]] = attr

for i in range(len(elements)):

    if flowX[i] != '':
        pairs.append((elements[i], flowX[i]))
        attr = {'category': 'flowController'}
        attrdi[flowX[i]] = attr

for i in range(len(elements)):

    if flowT[i] != '':
        pairs.append((elements[i], flowT[i]))
        attr = {'category': 'flowTerminal'}
        attrdi[flowT[i]] = attr

for i in range(len(elements)):

    if flowS[i] != '':
        pairs.append((elements[i], flowS[i]))
        attr = {'category': 'flowSegment'}
        attrdi[flowS[i]] = attr

for i in range(len(elements)):

    if flowF[i] != '':
        pairs.append((elements[i], flowF[i]))
        attr = {'category': 'flowFitting'}
        attrdi[flowF[i]] = attr

for i in range(len(elements)):

    if window[i] != '':
        pairs.append((elements[i], window[i]))
        attr = {'category': 'window'}
        attrdi[window[i]] = attr

for i in range(len(elements)):

    if column[i] != '':
        pairs.append((elements[i], column[i]))
        attr = {'category': 'column'}
        attrdi[column[i]] = attr

for i in range(len(elements)):

    if dist[i] != '':
        pairs.append((elements[i], dist[i]))
        attr = {'category': 'distribution_control_element'}
        attrdi[dist[i]] = attr

for i in range(len(elements)):

    if stair[i] != '':
        pairs.append((elements[i], stair[i]))
        attr = {'category': 'stair'}
        attrdi[stair[i]] = attr

### Making networkX graphs

G = nx.Graph()
fig, ax = plt.subplots()

nodes = list(elements)
G.add_nodes_from(nodes)
G.add_edges_from(pairs)
pos = nx.nx_agraph.graphviz_layout(G) ## this takes the positions of nodes


nx.set_node_attributes(G, attrdi)



## colors

groups = set(nx.get_node_attributes(G,'category').values())
mapping = dict(zip(sorted(groups),count()))

nodes = G.nodes()
colors = []
print(mapping)

for n in G.nodes():
        if bool(G.nodes[n]):

            colors.append(mapping[G.nodes[n]['category']])

        else:
            colors.append('5')
        #print(n)

print(colors)
#colors = { 0 : 'red', 1: 'blue' , 2: 'green', 3: 'black'}
nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.2)
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


plt.draw()
plt.show()
