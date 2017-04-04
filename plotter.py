import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def hist_subplot(subplot, data, title):
    plt.subplot(subplot)
    plt.title(title)
    plt.hist(data, normed=False, bins=20)
    axes = plt.gca()
    axes.set_xlim([0,600])
    axes.set_ylim([0,10])
    plt.ylabel('Frequency')
    plt.xlabel('Generations')
    plt.grid(True)


try: 
    f = open('./result/result.pickle')
    result = pickle.load(f)
    f.close()

    pathnet_first = []
    pathnet_second = []
    control_first = []
    control_second = []

    for res in result:
        if res[0] == 'pathnet':
            pathnet_first.append(res[1])
            pathnet_second.append(res[2])
        else:
            control_first.append(res[1])
            control_second.append(res[2])

    pathnet = [first + second for first, second in zip(pathnet_first, pathnet_second)]
    control = [first + second for first, second in zip(control_first, control_second)]

    hist_subplot('231', pathnet_first, 'pathnet task 1')
    hist_subplot('232', pathnet_second, 'pathnet task 2')
    hist_subplot('233', pathnet, 'pathnet total')
    hist_subplot('234', control_first, 'control task 1')
    hist_subplot('235', control_second, 'control task 2')
    hist_subplot('236', control, 'control total')

    plt.show()

except IOError:
    print("Result file not exist")
