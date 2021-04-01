import h5py

def store_data_in_hdffile(name_, data, hf, start, end, Total_no_sample=10000):
    if (name_ not in hf):
        hf.create_dataset(name_, (np.append(Total_no_sample, data[0].shape)),
                          'float64')

    hf[name_][start:end] = data