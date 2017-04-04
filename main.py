import os
import argparse
import random
import pickle

import torch
from torchvision import datasets, transforms

import pathnet
import genotype
import mnist_dataset
import visualize

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--num-batch', type=int, default=50, metavar='N',
                    help='input batch number for each episode (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of maximum epochs to train (default: 300)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')##0.01
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--module-num', type=int, default=[10,10,10], metavar='N',
                    help='number of modules in each layer')
parser.add_argument('--neuron-num', type=int, default=20, metavar='N',
                    help='number of modules in each layer')
parser.add_argument('--noise-prob', type=float, default=0.5, metavar='N',
                    help='salt and pepper noise rate')
parser.add_argument('--success-threshold', type=float, default=0.998, metavar='N',
                    help='accuracy threshold to finish the first task')
parser.add_argument('--readout-num', type=int, default=2, metavar='N',
                    help='number of units for readout (default: 2 for MNIST binary classification task)')
parser.add_argument('--control', action='store_true', default=False,
                    help='controlled experiment on/off')
parser.add_argument('--no-visualize', dest='vis', action='store_false', default=True,
                    help='show graph')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
def train_pathnet(model, gene, visualizer, train_loader, best_fitness, best_path, epoch, vis_color):
    pathways = gene.sample()
    fitnesses = []
    train_data = [(data, target) for (data,target) in train_loader]
    for pathway in pathways:
        path = pathway.return_genotype()
        fitness = model.train_model(train_data, path, args.num_batch)
        fitnesses.append(fitness)
    print("Epoch {} : Fitnesses = {} vs {}".format(epoch, fitnesses[0], fitnesses[1]))
    gene.overwrite(pathways, fitnesses)
    genes = gene.return_all_genotypes()
    visualizer.show(genes, vis_color)
    if max(fitnesses) > best_fitness:
        best_fitness = max(fitnesses)
        best_path = pathways[fitnesses.index(max(fitnesses))].return_genotype()
    return best_fitness, best_path

def train_control(model, gene, visualizer, train_loader, epoch):        
    path = gene.return_control_genotype()
    train_data = [(data, target) for (data,target) in train_loader]
    fitness = model.train_model(train_data, path, args.num_batch)
    print("Epoch {} : Fitness = {}".format(epoch, fitness))
    genes = [gene.return_control_genotype()] * 64
    visualizer.show(genes, 'm')
    return fitness

def main():
    model = pathnet.Net(args)
    gene = genotype.Genetic(3, 10, 3, 64)
    visualizer = visualize.GraphVisualize(args.module_num, args.vis)
    
    if args.cuda:
        model.cuda()
    
    if not os.path.isdir('./data/mnist'):
        os.system('./get_mnist_data.sh')

    if not os.path.isdir('./result'):
        os.makedirs("./result")
    if os.path.exists('./result/result.pickle'):
        f = open('./result/result.pickle','r')
        result = pickle.load(f)
        f.close()
    else:
        result = []

    prob = args.noise_prob
    dataset = mnist_dataset.Dataset(prob)
    labels = random.sample(range(10), 2)
    labels = [4,5]
    print("Two training classes : {} and {}".format(labels[0], labels[1]))
    dataset.set_binary_class(labels[0], labels[1])
    train_loader = dataset.convert2tensor(args)

    """first task"""
    print("First task started...")
    best_fitness = 0.0
    best_path = [[None] * 3] * 3
    epoch = 0
    while True:
        epoch += 1
        if not args.control:
            best_fitness, best_path = train_pathnet(model, gene, visualizer, train_loader, best_fitness, best_path, epoch, 'm')
            if best_fitness > args.success_threshold:
                print("Fitness achieved accuracy threshold!! Move to next task...")
                break

        else: ##control experiment
            fitness = train_control(model, gene, visualizer, train_loader, epoch)
            if fitness > args.success_threshold:
                print("Fitness achieved accuracy threshold!! Move to next task...")
                break

            #best_path = path
    first_epoch = epoch
    visualizer.set_fixed(best_path, 'r')

    print("Second task started...")

    if not args.control:
        gene = genotype.Genetic(3, 10, 3, 64)
        model.init(best_path)
        #model.init()
    else:
        model.init()##de novo
        #pass##findtune

    print("Two training classes : {} and {}".format(labels[0], labels[1]))

    labels = random.sample(range(10), 2)
    labels = [4,5]
    dataset.set_binary_class(labels[0], labels[1])
    train_loader = dataset.convert2tensor(args)

    best_fitness = 0.0    
    best_path = [[None] * 3] * 3
    for epoch in range(1 + first_epoch, args.epochs + 1 + first_epoch):
        if not args.control:
            best_fitness, best_path = train_pathnet(model, gene, visualizer, train_loader, best_fitness, best_path, epoch, 'c')
            if best_fitness > args.success_threshold:
                print("Fitness achieved accuracy threshold!! Move to next task...")
                break

        else: ##control experiment
            fitness = train_control(model, gene, visualizer, train_loader, epoch)
            if fitness > args.success_threshold:
                print("Fitness achieved accuracy threshold!! Move to next task...")
                break

    second_epoch = epoch - first_epoch
    print "first {}, second{}".format(first_epoch, second_epoch)

    if args.control:
        result.append(('control', first_epoch, second_epoch))
    else:
        result.append(('pathnet', first_epoch, second_epoch))

    f = open('./result/result.pickle', 'w')
    pickle.dump(result, f)
    f.close()

if __name__ == '__main__':
    main()
