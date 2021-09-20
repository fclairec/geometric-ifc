from torch_geometric import data
import trimesh
import glob
import pandas as pd
import os
import csv

dataset = glob.glob('/home/asalman/ifcwork/ifcworkspi2021/resources/BIMGEOMV1/raw/**/*.ply', recursive = True)
volume = []
for ply in dataset:
    mesh = trimesh.load(ply)
    volume.append(mesh.volume) 

print(volume)

df = pd.DataFrame(list(zip(*[dataset, volume]))).add_prefix('Col')

df.to_csv('file.csv', index=False)




