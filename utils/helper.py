import h5py
import numpy as np


def store_data_in_hdffile(name_, data, hf, start, end, Total_no_sample=10000):
    if (name_ not in hf):
        hf.create_dataset(name_, (np.append(Total_no_sample, data[0].shape)),
                          'float64')

    hf[name_][start:end] = data


def scale01(M):

    New_M = np.zeros((M.shape))
    for i in range(M.shape[0]):
        for j in range(M.shape[3]):
            M_ = M[i, :, :, j]
            MIN = np.min(M_)
            MAX = np.max(M_)
            New_M[i, :, :, j] = 0.0 * M_ if (MAX == MIN) else (M_ - MIN) / (MAX - MIN)
    return New_M


def scale01_(M):

    New_M = np.zeros((M.shape))
    for i in range(M.shape[0]):
        for j in range(M.shape[3]):
            M_ = M[i, :, :, j]
            MIN = np.min(M_)
            MAX = np.max(M_)
            New_M[i, :, :, j] = 1.0 * M_ if (MAX == MIN) else (M_ - MIN) / (MAX - MIN)
    return New_M
