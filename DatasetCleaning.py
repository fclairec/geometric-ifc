import glob 
import os
import trimesh
import open3d as o3d
import string

dataset = glob.glob('/home/asalman/ifcwork/ifcworkspi2021/resources/BIMGEOMV1/raw/**/*.ply', recursive=True)
dirpath = '/home/asalman/ifcwork/ifcworkspi2021/resources/BIMGEOMV1/raw'
output_path = '/home/asalman/ifcwork/ifcworkspi2021/resources/'
output_path = os.path.join(output_path, 'BIMGEOMV2', 'raw')

def create_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
    print("Created Directory : ", dir)
  else:
    print("Directory already existed : ", dir)
  return dir


create_dir(output_path)




labels = ['IfcFlowTerminal', 'IfcFlowFitting' , 
            'IfcSlab', 'IfcFlowController','IfcDoor', 'IfcWindow', 'IfcFlowSegment',
            'IfcRailing', 'IfcWall', 'IfcStair', 'IfcColumn', 
            'IfcDistributionControlElement',  
            'IfcFurnishingElement' ]

for x in labels:

    output_path2 = os.path.join(output_path, x)
    train_path = os.path.join(output_path2, 'train')
    test_path = os.path.join(output_path2, 'test')
    create_dir(output_path2)
    create_dir(test_path)
    create_dir(train_path)

for dirpath, dirnames, files in os.walk(dirpath):
    #print(dirpath)
    for file_name in files:

        file = os.path.join(dirpath , file_name)
        mesh = trimesh.load(file)
        #mesh = o3d.io.read_triangle_mesh(mesh)
        trimesh.repair.fix_normals(mesh, multibody=False)
        
        print(file)
        
        dirpath2 = dirpath.replace("BIMGEOMV1","BIMGEOMV2")
        os.chdir(dirpath2)
        file_name2 = os.path.join(dirpath2, file_name)
        mesh.export(file_name2)
        #o3d.io.write_triangle_mesh(file_name, mesh, write_ascii=True )
        #trimesh.exchange.ply.export_ply(mesh, encoding='ascii')
        
