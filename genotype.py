import random
import numpy as np
import copy

class Genotype():

    def __init__(self, L, M, N):
        self.genotype = np.random.randint(0, M, (L,N))
        self.L = L
        self.M = M
        self.N = N

    def apply_mutation(self, i, j):
        gene = self.genotype[i][j] + random.randint(-2, 2)
        if gene < 0:
            gene += self.M
        elif gene > self.M - 1:
            gene -= self.M
        self.genotype[i][j] = gene

    def mutate(self):
        for i in range(self.L):
            for j in range(self.N):
                if random.random() < 1.0 / (self.L * self.N):
                    self.apply_mutation(i, j)

    def return_genotype(self):
        return self.genotype

    def overwrite(self, genotype):
        self.genotype = copy.deepcopy(genotype)


class Genetic():

    def __init__(self, L, M, N, pop): 
        """L: layers, M: units in each layer, N: number of active units, pop: number of gene"""
        self.genotypes = [Genotype(L, M, N) for _ in range(pop)]
        self.pop = pop
        self.control_fixed = random.sample(self.genotypes,1)[0]

    def return_all_genotypes(self):
        genotypes = [gene.return_genotype() for gene in self.genotypes]
        return genotypes

    def return_control(self):
        return self.control_fixed

    def return_control_genotype(self):
        return self.control_fixed.return_genotype()

    def sample(self):
        return random.sample(self.genotypes, 2)

    def overwrite(self, genotypes, fitnesses):
        win = genotypes[fitnesses.index(max(fitnesses))]
        lose = genotypes[fitnesses.index(min(fitnesses))]
        genotype = win.return_genotype()
        lose.overwrite(genotype)
        lose.mutate()
