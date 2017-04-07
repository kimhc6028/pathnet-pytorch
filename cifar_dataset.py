import numpy as np
import random
import cPickle as pickle

import torch
import torch.utils.data as data_utils

import common

class Dataset():
    
    def __init__(self, args):
        """load cifar dataset"""
        print("Loading cifar dataset...")
        train_data = self.unpickle('./data/cifar/data_batch_{}'.format(random.randint(1,5)))
        test_data = self.unpickle('./data/cifar/test_batch')
        self.train_loader = self.convert2tensor(train_data, args.batch_size, args.trainset_limit)
        self.test_loader = self.convert2tensor(test_data, args.batch_size, args.testset_limit)

    def unpickle(self, filename):
        fo = open(filename, 'rb')
        dict = pickle.load(fo)
        fo.close()
        return dict

    def convert2tensor(self, dataset, batch_size, limit):
        data = dataset['data']
        data = data[:limit]
        print("normalizing images...")
        data = common.normalize(data)
        print("done")
        target = dataset['labels']
        target = target[:limit]
        target = np.asarray(target)

        tensor_data = torch.from_numpy(data)
        tensor_data = tensor_data.float()
        tensor_target = torch.from_numpy(target)

        loader = data_utils.TensorDataset(tensor_data, tensor_target)
        loader_dataset = data_utils.DataLoader(loader, batch_size=batch_size, shuffle = True)
        return loader_dataset

    def return_dataset(self):
        return self.train_loader, self.test_loader

