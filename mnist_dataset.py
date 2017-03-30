from random import shuffle, sample

import numpy as np
import torch
import torch.utils.data as data_utils

from mnist import MNIST
from common import salt_and_pepper, normalize


class Dataset():

    def __init__(self, prob):
        """load mnist dataset"""
        mndata = MNIST('./data/mnist/raw')

        mnist_train_images, mnist_train_labels = mndata.load_training()
        mnist_test_images, mnist_test_labels = mndata.load_testing()

        mnist_train_images = np.asarray(mnist_train_images)
        mnist_train_images = normalize(mnist_train_images)
        mnist_train_labels = np.asarray(mnist_train_labels)

        mnist_test_images = np.asarray(mnist_test_images)
        mnist_test_images = normalize(mnist_test_images)
        mnist_test_labels = np.asarray(mnist_test_labels)

        """divide datset by label"""
        print("Dividing dataset...")
        sorted_train_images = []
        sorted_train_labels = []        
        sorted_test_images = []
        sorted_test_labels = []

        for label in range(0,10):
            train_index = np.where(mnist_train_labels == label)
            test_index = np.where(mnist_test_labels == label)
            sorted_train_images.append(mnist_train_images[train_index[0]])
            sorted_train_labels.append(np.asarray([label] * len(train_index[0])))
            sorted_test_images.append(mnist_test_images[test_index[0]])
            sorted_test_labels.append(np.asarray([label] * len(test_index[0])))
            
        """add salt_and_pepper noise"""
        print("Adding salt and pepper noise...")
        shape = 28 * 28 ##image shape of mnist
        self.train_images = []
        self.train_labels = sorted_train_labels
        self.test_images = []
        self.test_labels = sorted_test_labels
        for images in sorted_train_images:
            noise_images = []
            for image in images:
                noise_image = salt_and_pepper(image, prob, shape)
                noise_images.append(noise_image)
            self.train_images.append(noise_images)
        for images in sorted_test_images:
            noise_images = []
            for image in images:
                noise_image = salt_and_pepper(image, prob, shape)
                noise_images.append(noise_image)
            self.test_images.append(noise_images)

    def set_binary_class(self, label_1, label_2):
        """set which classes to train
        i.e., if class 1 and 8 are set, dataset only returns images with label 1 or 8"""
        train_first = [(image, label) for image, label in zip(self.train_images[label_1], self.train_labels[label_1])]

        train_second = [(image, label) for image, label in zip(self.train_images[label_2], self.train_labels[label_2])]

        self.binary_train_dataset = train_first + train_second
        self.binary_train_dataset = np.asarray(self.binary_train_dataset)
        shuffle(self.binary_train_dataset)
        
        test_first = [(image, label) for image, label in zip(self.test_images[label_1], self.test_labels[label_1])]
        test_second = [(image, label) for image, label in zip(self.test_images[label_2], self.test_labels[label_2])]
        self.binary_test_dataset = test_first + test_second
        shuffle(self.binary_test_dataset)

    def convert2tensor(self, args):
        data = np.asarray([e[0] for e in self.binary_train_dataset])
        target = np.asarray([e[1] for e in self.binary_train_dataset])

        tensor_data = torch.from_numpy(data)
        tensor_data = tensor_data.float()
        tensor_target = torch.from_numpy(target)

        train = data_utils.TensorDataset(tensor_data, tensor_target)
        train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle = True)
        return train_loader
