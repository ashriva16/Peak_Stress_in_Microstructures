from torch.autograd import Variable
import h5py
from filters import scale01, scale01_

class Microstructure_Data_Loader():

    def __init__():

        # Default Hdffile for reading data
        self.file = '../data/grain_data.h5'

        self._read_input_data()
        self._read_ground_truth_data()
        self._split_train_test()

    def _read_input_data(self,file):

        with h5py.File(file, 'r') as f:
            # With grain boundary
            # input_img = f['preprocess/input/grad_norm'][:]  # 40000
            input_img = f['input/grad_norm'][:]  # 10000

            input_img = input_img.reshape(input_img.shape[0],
                                        input_img.shape[1],
                                        input_img.shape[2], 1)
            self.input = scale01_(np.concatenate(
                (f['input/Eulerimages'], input_img), axis=3))  # both 40k and 10k

            # No grainboundary both 40k and 10k
            # input_img = scale01_(f['input/Eulerimages'])

    def _read_ground_truth_data(self,file):
        
        with h5py.File(file, 'r') as f:

            # Both 40k and 10k 32x32
            # self.ground_truth = scale01_(f['preprocess/vonmisses/vonmisses_32X32'])

            # 128X128
            self.ground_truth = scale01_(f['output/vonmisses/image'])  # 10000
            # self.ground_truth = scale01_(f['output/vonmisses2D'])  # 40000

    def _split_train_test(self):
        with h5py.File('../../data/split_indices.h5', 'r') as hf:
            indices_train = hf["train_indices"][:]
            indices_test = hf["test_indices"][:]

        self.x_train_val = self.input[indices_train]
        self.x_test = self.input[indices_test]

        # To save memory original data needs to be deleted
        del self.input

        self.y_train_val = self.ground_truth[indices_train]
        self.y_test = self.ground_truth[indices_test]

        # To save memory rest of the original data needs to be deleted
        del self.ground_truth, indices_train, indices_test

        print("\nreading done\n", flush=True)

    def _get_test_data(self):
        # Defining test pytorch tensors
        x_test = Variable(torch.from_numpy(self.x_test).transpose_(3, 1)).float().to(device)
        y_test = Variable(torch.from_numpy(self.y_test).transpose_(3, 1)).float().to(device)

        self.test_loader = data_utils.TensorDataset(x_test, y_test)

    def _get_train_data(self):
        # Defining train pytorch tensors
        x_train_val = Variable(torch.from_numpy(x_train_val).transpose_(3, 1)).float()
        y_train_val = Variable(torch.from_numpy(y_train_val).transpose_(3, 1)).float()
        
        train_val_loader = data_utils.TensorDataset(x_train_val, y_train_val)

    def get_data(self):

        self._get_test_data()
        self._get_train_data()

        return self.train_loader, self.test_loader