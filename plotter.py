import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--filename', type=str, default='result.pickle', metavar='N',
                    help='open result file')
args = parser.parse_args()



def subplot(subplot, data_first, data_second, title):
    plt.subplot(subplot)

    x = np.arange(0,100)
    y_first = np.mean(data_first, axis=0)
    y_second = np.mean(data_second, axis=0)
    y_first_err = np.std(data_first, axis=0) / 2.
    y_second_err = np.std(data_second, axis=0) / 2. 

    plt.fill_between(x, y_first - y_first_err, y_first + y_first_err, color='m', alpha=0.3)
    plt.fill_between(x, y_second - y_second_err, y_second + y_second_err, color='c', alpha=0.3)
    plt.plot(x, y_first, color='r', label='first task')
    plt.plot(x, y_second, color='g', label='second task')
    plt.legend(bbox_to_anchor=(1.0, 1.), loc=2, ncol=1, fontsize=15)
    axes = plt.gca()
    axes.set_xlim([0, 100])
    axes.set_ylim([0, 1.2])
    plt.title(title, fontsize=20, y = 0.9)
    plt.ylabel('Accuracy',fontsize=15)
    plt.xlabel('Generations',fontsize=15)
    plt.grid(True)


try: 
    f = open(os.path.join('./result', args.filename))
    result = pickle.load(f)
    f.close()

    pathnet_first = []
    pathnet_second = []

    for res in result:
        if res[0] == 'pathnet':
            pathnet_first.append(res[2])
            pathnet_second.append(res[3])

    #print pathnet_first, pathnet_second
    subplot('111', pathnet_first, pathnet_second,'pathnet')
    plt.show()

except IOError:
    print("Result file does not exist")
