from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import gradcheck

class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        best_path = [[None] * 3] * 3
        self.final_layers = []
        self.init(best_path)

    def init(self, best_path = [[None] * 3] * 3):
        neuron_num = self.args.neuron_num
        module_num = self.args.module_num

        """Initialize all parameters"""
        self.fc1 = []
        self.fc2 = []
        self.fc3 = []
        ##
        print best_path
        ##
        for i in range(module_num[0]):
            if not i in best_path[0]:
                """All parameters should be declared as member variable, so I think this is the simplest way to do so"""
                exec("self.m1" + str(i) + " = nn.Linear(28*28," + str(neuron_num) + ")")
            else:
                exec("print self.m1" + str(i))
            exec("self.fc1.append(self.m1" + str(i) + ")")

        for i in range(module_num[1]):
            if not i in best_path[1]:
                exec("self.m2" + str(i) + " = nn.Linear(" + str(neuron_num) + "," + str(neuron_num) + ")")
            else:
                exec("print self.m2" + str(i))
            exec("self.fc2.append(self.m2" + str(i) + ")")

        for i in range(module_num[2]):
            if not i in best_path[2]:
                #exec("self.m3" + str(i) + " = nn.Linear(" + str(neuron_num) + ", 10)")
                exec("self.m3" + str(i) + " = nn.Linear(" + str(neuron_num) + "," + str(neuron_num) + ")")
            else:
                exec("print self.m3" + str(i))
            exec("self.fc3.append(self.m3" + str(i) + ")")

        """final layer which is not inclued in pathnet. Independent for each task"""
        if len(self.final_layers) < 1:
            self.final_layer1 = nn.Linear(neuron_num, self.args.readout_num)
            self.final_layers.append(self.final_layer1)
        else:
            self.final_layer2 = nn.Linear(neuron_num, self.args.readout_num)
            self.final_layers.append(self.final_layer2)

        trainable_params = []
        params_set = [self.fc1, self.fc2, self.fc3]
        for path, params in zip(best_path, params_set):
            for i, param in enumerate(params):
                if  i in path:
                    param.requires_grad = False
                else:
                    p = {'params': param.parameters()}
                    trainable_params.append(p)
                    
        p = {'params': self.final_layers[-1].parameters()}
        trainable_params.append(p)
        self.optimizer = optim.Adam(trainable_params, lr=self.args.lr)
        if self.args.cuda:
            self.cuda()

    def forward(self, x, path, last):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1[path[0][0]](x)) + F.relu(self.fc1[path[0][1]](x)) + F.relu(self.fc1[path[0][2]](x))
        x = F.relu(self.fc2[path[1][0]](x)) + F.relu(self.fc2[path[1][1]](x)) + F.relu(self.fc2[path[1][2]](x))
        x = F.relu(self.fc3[path[2][0]](x)) + F.relu(self.fc3[path[2][1]](x)) + F.relu(self.fc3[path[2][2]](x))
        #x = self.final_layers[-1](x)
        x = self.final_layers[last](x)
        
        return x
        #return F.log_softmax(x)

    def train_model(self, train_loader, path, num_batch):
        self.train()
        fitness = 0
        train_len = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            #output = self(data, path)
            output = self(data, path, -1)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            fitness += pred.eq(target.data).cpu().sum()
            train_len += len(target.data)
            #loss = F.nll_loss(output, target)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            if not batch_idx < num_batch -1:
                break
        fitness = fitness / train_len
        return fitness

    def test_model(self, test_loader, path, last):
        self.eval()
        fitness = 0
        train_len = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self(data, path, last)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            fitness += pred.eq(target.data).cpu().sum()
            train_len += len(target.data)
            if batch_idx > 1000:
                break

        fitness = fitness / train_len
        return fitness
