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
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--module-num', type=int, default=[10,10,10], metavar='N',
                    help='number of modules in each layer')
parser.add_argument('--neuron-num', type=int, default=20, metavar='N',
                    help='number of modules in each layer')
parser.add_argument('--generation-limit', type=int, default=100, metavar='N',
                    help='number of generation to compute')
parser.add_argument('--noise-prob', type=float, default=0.5, metavar='N',
                    help='salt and pepper noise rate')
parser.add_argument('--threshold', type=float, default=0.998, metavar='N',
                    help='accuracy threshold to finish the first task')
parser.add_argument('--readout-num', type=int, default=2, metavar='N',
                    help='number of units for readout (default: 2 for MNIST binary classification task)')
parser.add_argument('--control', action='store_true', default=False,
                    help='controlled experiment on/off')
parser.add_argument('--fine-tune', action='store_true', default=False,
                    help='fine-tuning control experiment on/off')
parser.add_argument('--no-graph', dest='vis', action='store_false', default=True,
                    help='show graph')
parser.add_argument('--no-save', action='store_true', default=False,
                    help='do not save result')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
def train_pathnet(model, gene, visualizer, train_loader, best_fitness, best_path, gen, vis_color):
    pathways = gene.sample()
    fitnesses = []
    train_data = [(data, target) for (data,target) in train_loader]
    for pathway in pathways:
        path = pathway.return_genotype()
        fitness = model.train_model(train_data, path, args.num_batch)
        fitnesses.append(fitness)
    print("Generation {} : Fitnesses = {} vs {}".format(gen, fitnesses[0], fitnesses[1]))
    gene.overwrite(pathways, fitnesses)
    genes = gene.return_all_genotypes()
    visualizer.show(genes, vis_color)
    if max(fitnesses) > best_fitness:
        best_fitness = max(fitnesses)
        best_path = pathways[fitnesses.index(max(fitnesses))].return_genotype()
    return best_fitness, best_path, max(fitnesses)

def train_control(model, gene, visualizer, train_loader, gen):        
    path = gene.return_control_genotype()
    train_data = [(data, target) for (data,target) in train_loader]
    fitness = model.train_model(train_data, path, args.num_batch)
    print("Generation {} : Fitness = {}".format(gen, fitness))
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
    print("Two training classes : {} and {}".format(labels[0], labels[1]))
    dataset.set_binary_class(labels[0], labels[1])
    train_loader = dataset.convert2tensor(args)

    """first task"""
    print("First task started...")
    best_fitness = 0.0
    best_path = [[None] * 3] * 3
    gen = 0
    first_fitness = []

    for gen in range(args.generation_limit):
        if not args.control:
            best_fitness, best_path, max_fitness = train_pathnet(model, gene, visualizer, train_loader, best_fitness, best_path, gen, 'm')
            first_fitness.append(max_fitness)

        else: ##control experiment
            fitness = train_control(model, gene, visualizer, train_loader, gen)
            first_fitness.append(fitness)

    print("First task done!! Move to next task")
    print("Second task started...")

    if not args.control:
        #gene = genotype.Genetic(3, 10, 3, 64)
        model.init(best_path)
        visualizer.set_fixed(best_path, 'r')
    else:
        if not args.fine_tune:
            model = pathnet.Net(args)
            gene = genotype.Genetic(3, 10, 3, 64)

    c_1 = labels[0]

    while True:
        c_2 = random.randint(0, 10-1)
        if not c_2 == c_1:
            break
    labels = [c_1, c_2]

    print("Two training classes : {} and {}".format(labels[0], labels[1]))
    dataset.set_binary_class(labels[0], labels[1])
    train_loader = dataset.convert2tensor(args)

    best_fitness = 0.0    
    best_path = [[None] * 3] * 3
    gen = 0

    second_fitness = []
    for gen in range(args.generation_limit):
        if not args.control:
            best_fitness, best_path, max_fitness = train_pathnet(model, gene, visualizer, train_loader, best_fitness, best_path, gen, 'c')
            second_fitness.append(max_fitness)

        else: ##control experiment
            fitness = train_control(model, gene, visualizer, train_loader, gen)
            second_fitness.append(fitness)

    print("Second task done!! Goodbye!!")

    if not args.no_save:
        if args.control:
            if args.fine_tune:
                result.append(('fine_tune', args.threshold, first_fitness, second_fitness))
            else:
                result.append(('control', args.threshold, first_fitness, second_fitness))
        else:
            result.append(('pathnet', args.threshold, first_fitness, second_fitness))

        f = open('./result/result.pickle', 'w')
        pickle.dump(result, f)
        f.close()

if __name__ == '__main__':
    main()
