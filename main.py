import argparse
import torch
from torchvision import datasets, transforms
import random

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
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.0001)')##0.01
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--module_num', type=int, default=[10,10,10], metavar='N',
                    help='number of modules in each layer')
parser.add_argument('--neuron_num', type=int, default=20, metavar='N',
                    help='number of modules in each layer')
parser.add_argument('--noise_prob', type=float, default=0.5, metavar='N',
                    help='salt and pepper noise rate')
parser.add_argument('--success-threshold', type=float, default=0.998, metavar='N',
                    help='accuracy threshold to finish the first task')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    

def main():
    model = pathnet.Net(args)
    gene = genotype.Genetic(3, 10, 3, 64)
    visualizer = visualize.GraphVisualize(args.module_num)
    
    if args.cuda:
        model.cuda()

    prob = args.noise_prob 
    dataset = mnist_dataset.Dataset(prob)
    labels = random.sample(range(10), 2)
    print("Two training classes : {} and {}".format(labels[0], labels[1]))
    dataset.set_binary_class(labels[0], labels[1])
    train_loader = dataset.convert2tensor(args)

    """first task"""
    print("First task started...")
    best_fitness = 0.0
    best_genotype = None
    for epoch in range(1, args.epochs + 1):
        pathways = gene.sample()
        fitnesses = []
        for pathway in pathways:
            path = pathway.return_genotype()
            fitness = model.train_model(epoch, train_loader, path, args.num_batch)
            fitnesses.append(fitness)
        print("Epoch {} :Fitnesses = {} vs {}".format(epoch, fitnesses[0], fitnesses[1]))

        gene.overwrite(pathways, fitnesses)
        genes = gene.return_all_genotypes()
        visualizer.show(genes)
        last_epoch = epoch
        if max(fitnesses) > best_fitness:
            best_fitness = max(fitnesses)
            best_path = pathways[fitnesses.index(max(fitnesses))].return_genotype()
        if max(fitnesses) > args.success_threshold:
                print("Fitness achieved accuracy threshold!! Move to next task...")
                break

    """second task"""
    labels = random.sample(range(10), 2)
    print("Two training classes : {} and {}".format(labels[0], labels[1]))
    dataset.set_binary_class(labels[0], labels[1])
    train_loader = dataset.convert2tensor(args)

    print("Second task started...")
    model.re_init(best_path)
    for epoch in range(1 + last_epoch, args.epochs + 1 + last_epoch):
        pathways = gene.sample()
        fitnesses = []
        for pathway in pathways:
            path = pathway.return_genotype()
            fitness = model.train_model(epoch, train_loader, path, args.num_batch)
            fitnesses.append(fitness)
        print("Epoch {} :Fitnesses = {} vs {}".format(epoch, fitnesses[0], fitnesses[1]))
        gene.overwrite(pathways, fitnesses)
        genes = gene.return_all_genotypes()
        visualizer.show(genes)
        if max(fitnesses) > args.success_threshold:
                print("Fitness achieved accuracy threshold!! Goodbye!!")
                break


if __name__ == '__main__':
    main()
