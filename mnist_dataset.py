from random import shuffle, sample

import numpy as np
import torch
import torch.utils.data as data_utils

from mnist import MNIST
from common import salt_and_pepper, normalize


class Dataset():

    def __init__(self, prob):
        """load mnist dataset"""
        print("Loading MNIST dataset...")
        mndata = MNIST('./data/mnist/')

        mnist_train_images, mnist_train_labels = mndata.load_training()

        mnist_train_images = np.asarray(mnist_train_images)
        mnist_train_images = normalize(mnist_train_images)
        mnist_train_labels = np.asarray(mnist_train_labels)

        """divide datset by label"""
        print("Dividing dataset...")
        sorted_train_images = []
        sorted_train_labels = []        

        for label in range(0,10):
            train_index = np.where(mnist_train_labels == label)
            sorted_train_images.append(mnist_train_images[train_index[0]])
            sorted_train_labels.append(np.asarray([label] * len(train_index[0])))
            
        """add salt_and_pepper noise"""
        print("Adding salt and pepper noise...")
        shape = 28 * 28 ##image shape of mnist
        self.train_images = []
        self.train_labels = sorted_train_labels
        for images in sorted_train_images:
            noise_images = []
            for image in images:
                noise_image = salt_and_pepper(image, prob, shape)
                noise_images.append(noise_image)
            self.train_images.append(noise_images)

    def set_binary_class(self, label_0, label_1):
        """set which classes to train, and change their labels to binary.
        i.e., if class 1 and 8 are set, dataset only returns images with label 0 (1 -> 0) and 1 (8 -> 1)"""
        train_first = [(image, 0) for image in self.train_images[label_0]]
        train_second = [(image, 1) for image in self.train_images[label_1]]
        
        self.binary_train_dataset = np.asarray(train_first + train_second)
        #shuffle(self.binary_train_dataset)
        
    def convert2tensor(self, args):
        data = np.asarray([e[0] for e in self.binary_train_dataset])
        target = np.asarray([e[1] for e in self.binary_train_dataset])
        tensor_data = torch.from_numpy(data)
        tensor_data = tensor_data.float()
        tensor_target = torch.from_numpy(target)

        train = data_utils.TensorDataset(tensor_data, tensor_target)
        train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle = True)
        return train_loader
