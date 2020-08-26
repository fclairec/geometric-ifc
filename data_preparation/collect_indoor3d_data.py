import os
import sys
from data_preparation.indoor3d_util import collect_point_label, test, collect_point_label_nonnorm


def collect_indoor3d_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_PATH = os.path.join(ROOT_DIR, '../../AspernV1')

    """cloud_train = os.path.join(DATA_PATH, 'all_subsampled_train.txt')
    cloud_test = os.path.join(DATA_PATH, 'all_subsampled_test.txt')"""

    cloud_train_full = os.path.join(DATA_PATH, 'all-new-train.txt')
    cloud_test_full = os.path.join(DATA_PATH, 'all-new-test.txt')

    #cloud = os.path.join(DATA_PATH, 'all_subsampled.txt')


    """output_folder_small = os.path.join(DATA_PATH, 'asp_small')
    if not os.path.exists(output_folder_small):
        os.mkdir(output_folder_small)"""

    output_folder_full = os.path.join(DATA_PATH, 'asp_full')
    if not os.path.exists(output_folder_full):
        os.mkdir(output_folder_full)

    out_filename_train = 'asp_train.npy'
    out_filename_test = 'asp_test.npy'
    out_filename_inf = 'asp_inf.npy'


    """collect_point_label(cloud_train, os.path.join(output_folder_small, out_filename_train), 'numpy')
    test(os.path.join(output_folder_small, out_filename_train))

    collect_point_label(cloud_test, os.path.join(output_folder_small, out_filename_test), 'numpy')
    test(os.path.join(output_folder_small, out_filename_test))"""

    collect_point_label(cloud_train_full, os.path.join(output_folder_full, out_filename_train), 'numpy')
    test(os.path.join(output_folder_full, out_filename_train))

    collect_point_label(cloud_test_full, os.path.join(output_folder_full, out_filename_test), 'numpy')
    test(os.path.join(output_folder_full, out_filename_test))

    collect_point_label_nonnorm(cloud_test_full, os.path.join(output_folder_full, out_filename_test), 'numpy')
    test(os.path.join(output_folder_full, out_filename_test))


if __name__ == '__main__':
    collect_indoor3d_data()
