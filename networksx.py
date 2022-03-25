import networkx as nx
import matplotlib.pyplot as plt
import csv
import pandas as pd
from pandas import DataFrame 
import graphviz



# G = nx.petersen_graph()

# ax = plt.subplot()

# items = [range(5,10), range(5)]

# nx.draw_shell(G, nlist=items,with_labels=True, font_weight='bold')
# plt.show()
# items = range(5, 10), range(5)

# for item in items:
#     print(item)

file = '/home/asalman/ifcwork/ifcworkspi2021/geometric-ifc/sampledata.csv'
# with open(file, 'r') as f:

#     data = csv.reader(f)

#     for item in data:
#         print(item)

df = pd.read_csv(file)

Data = open(file, 'r')

# next(Data, None)
csv_reader = csv.reader(Data)

# for row in csv_reader:
#     print(row)
    
#     for item in row:
#         if item != 0:

            
elements = df['element'].to_list()
windows = df['window'].to_list()
slabs = df['slab'].to_list()
doors = df['door'].to_list()
walls = df['wall'].to_list()

pairs = []
for i in range(len(elements)):
    
    if windows[i] != 0:
        pairs.append((elements[i], windows[i]))

for i in range(len(elements)):
    
    if slabs[i] != 0:
        pairs.append((elements[i], slabs[i]))

for i in range(len(elements)):
    
    if doors[i] != 0:
        pairs.append((elements[i], doors[i]))

for i in range(len(elements)):
    
    if walls[i] != 0:
        pairs.append((elements[i], walls[i]))



# print(pairs)
G = nx.Graph()
pos = nx.nx_agraph.graphviz_layout(G)
fig,ax = plt.subplots()
G.add_edges_from(pairs)
attrs = {121:{'element':'door'},122:{'element':'slab'},123:{'element':'Window'},124:{'element':'wall'},125:{'element':'window'}}
nx.set_node_attributes(G, attrs)

nodes = nx.draw_networkx_nodes(G, pos)
##nx.draw_networkx(G)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    node = ind["ind"][0]
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


