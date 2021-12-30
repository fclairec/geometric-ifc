import torch
from torch_geometric.data import Data

try:
    import openmesh
except ImportError:
    openmesh = None


def read_ply(path):
    if openmesh is None:
        raise ImportError('`read_ply` requires the `openmesh` package.')

    mesh = openmesh.read_trimesh(path)
    pos = torch.from_numpy(mesh.points()).to(torch.float)
    #TODO norm
    face = torch.from_numpy(mesh.face_vertex_indices())
    face = face.t().to(torch.long).contiguous()
    return Data(pos=pos, face=face)

def read_ply_binary(path):
    if openmesh is None:
        raise ImportError('`read_ply` requires the `openmesh` package.')

    mesh = openmesh.read_trimesh(path, binary=True)
    pos = torch.from_numpy(mesh.points()).to(torch.float)
    #TODO norm
    face = torch.from_numpy(mesh.face_vertex_indices())
    face = face.t().to(torch.long).contiguous()
    return Data(pos=pos, face=face)

def read_ply_pcd(path):
    if openmesh is None:
        raise ImportError('`read_ply` requires the `openmesh` package.')

    mesh = openmesh.read_trimesh(path, binary=False, vertex_normal=True, vertex_color=True)
    pos = torch.from_numpy(mesh.points()).to(torch.float)
    norm = torch.from_numpy(mesh.vertex_normals()).to(torch.float)
    color = torch.from_numpy(mesh.vertex_colors()).to(torch.float)
    return Data(pos=pos, normal=norm, x=color)
