import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--mnist', action='store_true', default=False,
                    help='open mnist result')
args = parser.parse_args()


def subplot(subplot, data_first, data_second, title):
    plt.subplot(subplot)
    if args.mnist:
        x = np.arange(0,100)
    else:
        x = np.arange(0,500)
    y_first = np.mean(data_first, axis=0)
    y_second = np.mean(data_second, axis=0)
    y_first_err = np.std(data_first, axis=0) / 2.
    y_second_err = np.std(data_second, axis=0) / 2. 

    plt.fill_between(x, y_first - y_first_err, y_first + y_first_err, color='m', alpha=0.3)
    plt.fill_between(x, y_second - y_second_err, y_second + y_second_err, color='c', alpha=0.3)
    plt.plot(x, y_first, color='r', label='Task A')
    plt.plot(x, y_second, color='g', label='Task B (transfer learning)')
    plt.legend(bbox_to_anchor=(0.8, 0.3), loc=2, ncol=1, fontsize=15)
    axes = plt.gca()

    if args.mnist:
        axes.set_xlim([0, 100])
        axes.set_ylim([0, 1.2])
    else:
        axes.set_xlim([0, 500])
        axes.set_ylim([0, 0.6])
    plt.title(title, fontsize=20, y = 0.9)
    plt.ylabel('Accuracy',fontsize=15)
    plt.xlabel('Generations',fontsize=15)
    plt.grid(True)


try: 
    if args.mnist:
        f = open(os.path.join('./result/result_mnist.pickle'))
        result = pickle.load(f)
        f.close()
        pathnet_first = []
        pathnet_second = []

        for res in result:
            pathnet_first.append(res[2])
            pathnet_second.append(res[3])

        subplot('111', pathnet_first, pathnet_second,'MNIST')
        plt.show()


    else:
        f = open(os.path.join('./result/result_cifar_svhn.pickle'))
        result = pickle.load(f)
        f.close()

        cifar_first = []
        cifar_second = []
        svhn_first = []
        svhn_second = []

        for res in result:
            if res[0] == 'pathnet_cifar_first':
                cifar_first.append(res[2])
                svhn_second.append(res[3])
            else:
                svhn_first.append(res[2])
                cifar_second.append(res[3])

    subplot('211', cifar_first, cifar_second,'CIFAR-10')
    subplot('212', svhn_first, svhn_second,'cSVHN')

    plt.show()

except IOError:
    print("Result file does not exist")
