# Geometric-ifc learning

This repository investigates methods of spatial graph convolutions for 3D Shape classification and Point cloud semantic segmentation. 

Academic datasets such as ModelNet10 and S3DIS are included as well as my own datasets. 

My own datasets come from 
1. BIM (Building Information Modelling) for classification tasks
2. Pointclouds scanned with NavVis M6 Scanners



- [Installation](#installation)
- [Dataset preparation](#preparation)
- [Shape classification](#shape-classification)
- [Point cloud semantic segmentation](#point-cloud-semantic-segmentation)


---
## Installation  
See pytorch geometric documentation 
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html


--- 
## Dataset proparation
In folder dataset_preparation:
- H5 file assembly for point cloud data
- batch definition

for dataset assembly of BIM object dataset please refer to other project submission


---
## Shape classification
Training and inference for classification:
- RUN python experiments.py

---
## Point cloud semantic segmentation

Training and inference for classification:
- RUN python experiments_seg.py



NOTE: 
Please configure your experiments before launch
