import os

import sys
import json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT_DIR, '../../AspernV1')
sys.path.append(BASE_DIR)
from data_preparation import data_prep_util
from data_preparation.indoor3d_util import room2blocks_wrapper_normalized
import numpy as np

# Constants

def gen_indoor3d_h5(data_label_files_in, train):

    data_dir = DATA_PATH
    indoor3d_data_dir = os.path.join(data_dir, 'asp')
    NUM_POINT = 4096
    H5_BATCH_SIZE = 1000
    data_dim = [NUM_POINT, 9]
    label_dim = [NUM_POINT]
    data_dtype = 'float32'
    label_dtype = 'uint8'

    # Set paths


    data_label_files = [os.path.join(indoor3d_data_dir, i) for i in data_label_files_in]
    output_dir = os.path.join(data_dir, '..', 'ASPERN', 'raw')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
    #output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
    output_all_file = os.path.join(output_dir, 'all_files' + train + '.txt')
    #fout_room = open(output_room_filelist, 'w')
    all_file = open(output_all_file, 'w')

    # --------------------------------------
    # ----- BATCH WRITE TO HDF5 -----
    # --------------------------------------
    batch_data_dim = [H5_BATCH_SIZE] + data_dim
    batch_label_dim = [H5_BATCH_SIZE] + label_dim
    h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
    h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
    buffer_size = 0  # state: record how many samples are currently in buffer
    h5_index = 0 # state: the next h5 file to save

    def insert_batch(h5_batch_data, h5_batch_label, buffer_size, h5_index, data, label, last_batch=False):

        #global buffer_size, h5_index
        data_size = data.shape[0]
        # If there is enough space, just insert
        if buffer_size + data_size <= h5_batch_data.shape[0]:
            h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
            h5_batch_label[buffer_size:buffer_size+data_size] = label
            buffer_size += data_size
        else: # not enough space
            capacity = h5_batch_data.shape[0] - buffer_size
            assert(capacity>=0)
            if capacity > 0:
               h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...]
               h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...]
            # Save batch data and label to h5 file, reset buffer_size
            h5_filename =  output_filename_prefix + '_' + str(h5_index) + train + '.h5'
            data_prep_util.save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
            print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
            h5_index += 1
            buffer_size = 0
            # recursive call
            insert_batch(h5_batch_data, h5_batch_label, buffer_size, h5_index, data[capacity:, ...], label[capacity:, ...], last_batch)
        if last_batch and buffer_size > 0:
            h5_filename =  output_filename_prefix + '_' + str(h5_index) + train + '.h5'
            data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
            print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
            h5_index += 1
            buffer_size = 0
        return h5_batch_data, h5_batch_label, buffer_size, h5_index


    sample_cnt = 0
    for i, data_label_filename in enumerate(data_label_files):
        #/home/fcollins/Dokumente/seg_dgcnn/data/stanford_indoor3d/Area_1_hallway_6.npy
        print(data_label_filename)
        data, label = room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=1.0, stride=1,
                                                     random_sample=False, sample_num=None)

        print('data.shape {0}, label.shape {1}'.format(data.shape, label.shape))
        """for _ in range(data.shape[0]):
            fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')
"""
        sample_cnt += data.shape[0]
        h5_batch_data, h5_batch_label, buffer_size, h5_index = insert_batch(h5_batch_data, h5_batch_label, buffer_size, h5_index, data, label, i == len(data_label_files)-1)

    """fout_room.close()"""
    print("Total samples: {0}".format(sample_cnt))

    print("h5_index")
    print(h5_index)

    for i in range(h5_index):
        all_file.write(os.path.join('ASPERN', 'raw', 'ply_data_all_') + str(i) +'.h5\n')
    all_file.close()

if __name__ == '__main__':
    gen_indoor3d_h5(data_label_files_in = ['asp_train_subsampled.npy'], train='train')
    gen_indoor3d_h5(data_label_files_in = ['asp_test_subsampled.npy'], train='test')