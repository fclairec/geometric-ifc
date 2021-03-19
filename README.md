# Geometric-ifc learning

This repository investigates methods of spatial graph convolutions for 3D Shape classification. 

Academic datasets such as ModelNet10 dataset is included as well as a self-assembled dataset of IFC geometries.
 
![alt text](https://github.com/fclairec/geometric-ifc/blob/master/resources/BIMGEOM.PNG?raw=true)



- [Installation](#installation)
- [Dataset preparation](#preparation)
- [Shape classification](#shape-classification)
- [Point cloud semantic segmentation](#point-cloud-semantic-segmentation)


---
## Installation  

Check out the Dockerfile for the required packages to run this code

For specific installations regarding pytorch geometric please refer to the library's documentation 
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html




---
## Shape classification
In experiments.py parameters are configured for training Several values can be input and all possible input configurations are run.
Check the first lines of experiments.py for the default values. 

- RUN python geometric-ifc/experiments.py --batch_size 32 42 --learning_rate=0.01 0.001 --samplePoints 1024 --rotation [0,0,180] [180,180,180] --model PN2NET GCNConv GCNPool

During training the following output is saved:
 
 - best performing model is saved as model_state_best_val.pth.tar 
 - a class report of the final model evaluated on the test set 
 - epoch losses and accuracies 
 - class confusion matrix 
 - and a file showing the parameter configuration of this specific run
 
 TODO: 
 Output final embedding space for similarity analysis
 
 
 In evaluate.py the following will be available soon 
- Class similarity analysis
- 




