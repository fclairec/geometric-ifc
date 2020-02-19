__author__='fiona.collins'

import torch


class SamplePoints(object):

    r"""
    Uniformly saples points from mesh faces according to their face area

    Args:
        num (init): The number of points to sample
        remove_faces (bool, optional): Ifc set to False, the face tensor will not be removed. Default option "true"
        include_normals (bool, optional): If set to True the normals for each sampled point are computed. Default option "false"
        generator_colors (bool, optional): Ifc set to true an artificial color representation is made. Default "False"

    """

    def __init__(self, num, remove_faces=True, include_normals=False, generate_colors=False):
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals
        self.generate_colors = generate_colors

    def __call__(self, data):
        pos, face = data.pos, data.face


    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)
