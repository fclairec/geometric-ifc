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
volume = df['Volume'].to_list()



pairs = []
attrdi = {}
nodes = []

categoriesname = ['wall','stair','slab','furnishing', 'flowT', 'flowS','flowF','flowX', 'dist', 'column' , 'window', 'door']
categories = [wall,stair,slab,furnishing, flowT, flowS,flowF,flowX, dist, column , window, door]

for i in range(len(elements)):
    counter = 0
    attr2 = {'volume': volume[i]}
    for cat in categories:
        if cat[i] != '':
            #print(cat[i])
            pairs.append((elements[i], cat[i]))
            attr = {'category': categoriesname[counter],'volume' : volume[i]}
            attrdi[cat[i]] = attr
            
            nodes.append(cat[i])
        counter += 1

# for i in range(len(elements)):
#     attr = {'volume' : volume[i]}
#     attrdi[elements[i]].update(attr)
#     print(attrdi[elements[i]])

# # looping through each element to make edges. This can be improved in terms of implementation
# for i in range(len(elements)):

#     if wall[i] != '':
#         pairs.append((elements[i], wall[i]))
#         attr = {'category': 'wall'}
#         attrdi[wall[i]] = attr
#         nodes.append(wall[i])

# for i in range(len(elements)):

#     if slab[i] != '':
#         pairs.append((elements[i], slab[i]))
#         attr = {'category': 'slab'}
#         attrdi[slab[i]] = attr
#         nodes.append(slab[i])

# for i in range(len(elements)):

#     if door[i] != '':
#         pairs.append((elements[i], door[i]))
#         attr = {'category': 'door'}
#         attrdi[door[i]] = attr
#         nodes.append(door[i])

# for i in range(len(elements)):

#     if furnishing[i] != '':
#         pairs.append((elements[i], furnishing[i]))
#         attr = {'category': 'furnishing_element'}
#         attrdi[furnishing[i]] = attr
#         nodes.append(furnishing[i])

# for i in range(len(elements)):

#     if flowX[i] != '':
#         pairs.append((elements[i], flowX[i]))
#         attr = {'category': 'flowController'}
#         attrdi[flowX[i]] = attr
#         nodes.append(flowX[i])

# for i in range(len(elements)):

#     if flowT[i] != '':
#         pairs.append((elements[i], flowT[i]))
#         attr = {'category': 'flowTerminal'}
#         attrdi[flowT[i]] = attr
#         nodes.append(flowT[i])

# for i in range(len(elements)):

#     if flowS[i] != '':
#         pairs.append((elements[i], flowS[i]))
#         attr = {'category': 'flowSegment'}
#         attrdi[flowS[i]] = attr
#         nodes.append(flowS[i])

# for i in range(len(elements)):

#     if flowF[i] != '':
#         pairs.append((elements[i], flowF[i]))
#         attr = {'category': 'flowFitting'}
#         attrdi[flowF[i]] = attr
#         nodes.append(flowF[i])

# for i in range(len(elements)):

#     if window[i] != '':
#         pairs.append((elements[i], window[i]))
#         attr = {'category': 'window'}
#         attrdi[window[i]] = attr
#         nodes.append(window[i])

# for i in range(len(elements)):

#     if column[i] != '':
#         pairs.append((elements[i], column[i]))
#         attr = {'category': 'column'}
#         attrdi[column[i]] = attr
#         nodes.append(column[i])

# for i in range(len(elements)):

#     if dist[i] != '':
#         pairs.append((elements[i], dist[i]))
#         attr = {'category': 'distribution_control_element'}
#         attrdi[dist[i]] = attr
#         nodes.append(dist[i])

# for i in range(len(elements)):

#     if stair[i] != '':
#         pairs.append((elements[i], stair[i]))
#         attr = {'category': 'stair'}
#         attrdi[stair[i]] = attr
#         nodes.append(stair[i])


pairs = list(filter(lambda x: x[0] in nodes, pairs))

F = nx.Graph()
fig, ax = plt.subplots()

elementslist = list(elements)

for i in elementslist[:]:
    if i not in nodes:
        #print(i)
        elementslist.remove(i)



nodes = []
print(nodes)
nodes = elementslist




F.add_nodes_from(nodes)
F.add_edges_from(pairs)
F.add_edges_from(pairs)

pos = nx.nx_agraph.graphviz_layout(F) ## this takes the positions of nodes


nx.set_node_attributes(F, attrdi)


## colors



groups = set(nx.get_node_attributes(F,'category').values())
mapping = dict(zip(sorted(groups),count()))



nodes = F.nodes()
colors = []
#print(mapping)


for n in F.nodes():
        if bool(F.nodes[n]):

            colors.append(mapping[F.nodes[n]['category']])
            

        else:
            colors.append('5')
            #
            print(n)
        
        
        

#print(colors)
#colors = { 0 : 'red', 1: 'blue' , 2: 'green', 3: 'black'}
nx.draw_networkx_edges(F, pos=pos, ax=ax, alpha=0.3)
nodes = nx.draw_networkx_nodes(F, pos=pos, ax=ax, node_color=colors, cmap=plt.cm.jet)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

idx_to_node_dict = {}
for idx, node in enumerate(F.nodes):
    idx_to_node_dict[idx] = node

def update_annot(ind):
    node_idx = ind["ind"][0]
    node = idx_to_node_dict[node_idx]
    xy = pos[node]
    annot.xy = xy
    node_attr = {'node': node}
    node_attr.update(F.nodes[node])
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
