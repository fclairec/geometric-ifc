import os
import sys
from data_preparation.indoor3d_util import collect_point_label, test


def collect_indoor3d_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_PATH = os.path.join(ROOT_DIR, '../../AspernV1')

    cloud = os.path.join(DATA_PATH, 'all-1-with-colorcode.txt')
    #cloud = os.path.join(DATA_PATH, 'all_subsampled.txt')


    output_folder = os.path.join(DATA_PATH, 'asp')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    out_filename = 'asp_ug.npy'


    collect_point_label(cloud, os.path.join(output_folder, out_filename), 'numpy')
    test(os.path.join(output_folder, out_filename))


if __name__ == '__main__':
    collect_indoor3d_data()
