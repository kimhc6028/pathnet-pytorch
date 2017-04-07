import numpy as np
import scipy.io

import torch
import torch.utils.data as data_utils

import common


class Dataset():
    
    def __init__(self, args):
        """load cSVHN dataset"""
        print("Loading cSVHN dataset...")
        train_data = scipy.io.loadmat('./data/svhn/train_32x32.mat')
        test_data = scipy.io.loadmat('./data/svhn/test_32x32.mat')
        self.train_loader = self.convert2tensor(train_data, args.batch_size, args.trainset_limit)
        self.test_loader = self.convert2tensor(test_data, args.batch_size, args.testset_limit)
        
    def convert2tensor(self, dataset, batch_size, limit):
        b_data = dataset['X']
        b_data = b_data[:limit]
        print("normalizing images...")
        b_data = common.normalize(b_data)
        print("done")
        target = dataset['y']
        target = target.reshape((len(target)))
        target = target[:limit]
        """SVHN dataset is between 1 to 10: shift this to 0 to 9 to fit with neural network"""
        target = target - 1

        data = []
        for i in range(len(target)):
            data.append(b_data[:,:,:,i])
        data = np.asarray(data)
        tensor_data = torch.from_numpy(data)
        tensor_data = tensor_data.float()
        tensor_target = torch.from_numpy(target)

        loader = data_utils.TensorDataset(tensor_data, tensor_target)
        loader_dataset = data_utils.DataLoader(loader, batch_size=batch_size, shuffle = True)
        return loader_dataset

    def return_dataset(self):
        return self.train_loader, self.test_loader
