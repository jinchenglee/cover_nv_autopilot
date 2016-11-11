#!/usr/bin/env python3

import numpy as np
import dataset as d
import good_files as gf
import steer as steer

BATCH_SIZE = 50
N_ROUNDS_PER_DATASET = 5


def main():
    np.set_printoptions(precision=5, suppress=True)

    c2_net = steer.steer_nn()
    data = d.dataset()

    for file_num in gf.train_list:
        # Open dataset
        file_name = "/home/vitob/Downloads/deepdrive_hdf5/train_"+str(file_num).zfill(4)+".zlib.h5"
        print("Training on file " + file_name)
        data.open_dataset(file_name)

        #train over a signle dataset N rounds
        for i in range(int(data.num_images * N_ROUNDS_PER_DATASET /BATCH_SIZE)):

            xs, ys = data.LoadTrainBatch(BATCH_SIZE)
            loss = c2_net.train(xs, ys, i)
            print("train: step ", i, ", loss = ", loss)

            if i % 10 == 0:
                xs, ys = data.LoadValBatch(BATCH_SIZE)
                c2_net.val(xs, ys, i)

        # Save parameters
        c2_net.saveParam()

        # Close dataset
        data.close_dataset()

if __name__ == '__main__':
    main()

