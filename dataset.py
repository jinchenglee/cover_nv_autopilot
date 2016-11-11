import scipy.misc
import random
import tables
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

SCALE_ANGLE = 10

class dataset():
    """
    Data preparation for training/validation.
    """

    def __init__(self):
        self.ys = []
        
        #points to the end of the last batch
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0
        
    def open_dataset(self, file):
        self.f = tables.open_file(file,'r')
        self.cam = self.f.root.images
        self.ys = SCALE_ANGLE * self.f.root.targets[:,4]
        self.index = list(range(self.cam.shape[0]))
        self.train_idx, self.val_idx = train_test_split(self.index, test_size = 0.2)
        
        self.num_train_images = len(self.train_idx)
        self.num_val_images = len(self.val_idx)
        
        self.num_images = len(self.index)

    def close_dataset(self):
        self.f.close()

    def LoadTrainBatch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            img = self.cam[self.train_idx[(self.train_batch_pointer + i) % self.num_train_images],:,46:226,:]
            img = img.swapaxes(0,1)
            img = img.swapaxes(1,2)
            img = scipy.misc.imresize(img, [45, 160])
            tmp = np.zeros_like(img, dtype=np.float)
            for channel in range(3):
                img_ch_max = np.max(img[:,:,channel])
                img_ch_min = np.min(img[:,:,channel])
                tmp[:,:,channel] = (img[:,:,channel]-img_ch_min)/(img_ch_max - img_ch_min)
            x_out.append(tmp)
            #x_out.append(scipy.misc.imresize(img, [45, 160]) / 255.0)
            y_out.append([self.ys[self.train_idx[(self.train_batch_pointer + i) % self.num_train_images]]])
        self.train_batch_pointer += batch_size
        return x_out, y_out
    
    def LoadValBatch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            img = self.cam[self.train_idx[(self.val_batch_pointer + i) % self.num_val_images],:,46:226,:]
            img = img.swapaxes(0,1)
            img = img.swapaxes(1,2)
            img = scipy.misc.imresize(img, [45, 160])
            tmp = np.zeros_like(img, dtype=np.float)
            for channel in range(3):
                img_ch_max = np.max(img[:,:,channel])
                img_ch_min = np.min(img[:,:,channel])
                tmp[:,:,channel] = (img[:,:,channel]-img_ch_min)/(img_ch_max - img_ch_min)
            x_out.append(tmp)
            #x_out.append(scipy.misc.imresize(img, [45, 160]) / 255.0)
            y_out.append([self.ys[self.train_idx[(self.val_batch_pointer + i) % self.num_val_images]]])
        self.val_batch_pointer += batch_size
        return x_out, y_out
