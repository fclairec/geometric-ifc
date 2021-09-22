from sklearn import preprocessing
from torch_geometric import data
import trimesh
import glob
import pandas as pd
import os
import csv


path = '/home/asalman/ifcwork/ifcworkspi2021/resources/BIMGEOMV1/raw/'
# #dataset = glob.glob('/home/asalman/ifcwork/ifcworkspi2021/resources/BIMGEOMV1/raw/**/*.ply', recursive = True)
volume = []

labels = ['IfcFlowTerminal' ]
            #'IfcSlab', 'IfcFlowController','IfcDoor', 'IfcWindow', 'IfcFlowSegment',
            #'IfcRailing', 'IfcWall', 'IfcStair', 'IfcColumn', 
            #'IfcDistributionControlElement',  
            #'IfcFurnishingElement' ]


main_dataset = []
main_dataset2 = []
main_volume =[]

for x in labels:
    
    class_path = os.path.join(path, x, '**/*.ply')
    
    dataset = glob.glob(class_path, recursive= True)
    
    
    
    volume = []
    
    
    for ply in dataset:
          
        mesh = trimesh.load(ply)
        trimesh.repair.fix_normals(mesh, multibody=False)
        volume.append(mesh.volume)
        

    print ('#################################')
   
    normalized_volume = preprocessing.normalize([volume])

   
    print('Updating dataset')
    main_dataset.append(dataset)
    
    print('Updating dataset')
    main_volume.append(normalized_volume)
    




# normal = True

# for ply in dataset:
#     mesh = trimesh.load(ply)
#     if normal == True:
#         trimesh.repair.fix_normals(mesh, multibody=False)

#     volume.append(mesh.volume) 

print(main_dataset)
print(main_volume)
df = pd.DataFrame(list(zip(*[main_dataset, main_volume]))).add_prefix('Col')



# path = 
# for row in df:
#     if row[0] == os.path.join(path, labels[0], 
# # normalized_df = (df-df.mean())/df.std()

df.to_csv('file3.csv', index=False)
