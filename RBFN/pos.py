###########   ParticleSwarmOptimization   ###########
#   vid = w*vid+c1*rand()*(pid-xid)+c2*Rand()*(pgd-xid)
#   xid = xid+vid
#
#   Xi = (xi1, xi2, ... , xiD)
#   Pi = (pi1, pi2, ... , piD)
#   Vi = (vi1, vi2, ... , viD)

import numpy as np

class ParticleSwarmOptimization(object):
    def __init__(self, populationSize, c1, c2, hiddenLayerNeuralNumber):
        self.c1 = c1
        self.c2 = c2
        self.populationSize = int(populationSize)
        self.hiddenLayerNeuralNumber = hiddenLayerNeuralNumber
        self.vmax = 1

    #The x-vector records the current position (location) of the particle in the search space,
    #The p-vector records the location of the best solution found so far by the particle, and
    #The v-vector contains a gradient (direction) for which particle will travel in if undisturbed.
    #(theta, w1 ,w2 , …, wj , m11, m12, …, m1i, m21, m22, …, m2i, …, mj1, mj2,…, mji, σ1, σ2, …, σj)
    def create_particle_x_p_v(self, trainData):
        self.Dim = 1 + self.hiddenLayerNeuralNumber + 3*self.hiddenLayerNeuralNumber + self.hiddenLayerNeuralNumber
        self.particle = [np.random.uniform(-1.0, 1.0, size=(self.Dim+1,)) for _ in range(int(self.populationSize))]
        self.particle_x = np.array(self.particle)
        self.particle = [np.random.uniform(-1.0, 1.0, size=(self.Dim + 1,)) for _ in range(int(self.populationSize))]
        self.particle_v = np.array(self.particle)

        self.calculate_Fitness_Function(trainData)
        self.particle_p = np.array(self.particle_x)


    #
    def _fitness_Function(self, particle_x, trainData):
        sum = 0
        for i in range(self.hiddenLayerNeuralNumber+1):
            if i == 0:
                sum += particle_x[0]
            else:
                sum += particle_x[i] * self._gaussian_basis_function(trainData[:,:3], particle_x[1 + self.hiddenLayerNeuralNumber + (i-1)*3 : 1 + self.hiddenLayerNeuralNumber + (i-1)*3 + 3], \
                                                     particle_x[1 + self.hiddenLayerNeuralNumber + self.hiddenLayerNeuralNumber*3 + (i-1): 1 + self.hiddenLayerNeuralNumber + self.hiddenLayerNeuralNumber*3 + (i-1) + 1])
        fitness = (trainData[:,3]-sum)**2
        fitnesssum = np.sum(fitness, axis = 0)/2
        return fitnesssum
    #trainData=[none, 3], m=[1, 3], sigma[1, 3]
    def _gaussian_basis_function(self, trainData, m, sigma):
        phi = np.exp(-(trainData - m)**2/(2*(sigma)**2))
        phi = np.sum(phi, axis=1)
        return phi
    #
    def calculate_Fitness_Function(self, trainData):
        for i in range(len(self.particle_x)):
            self.particle_x[i][self.Dim] = self._fitness_Function(self.particle_x[i], trainData)
    #chose best solution found so far by the particle
    def chose_best_fitness_particle(self):
        for i in range(len(self.particle_p)):
            if self.particle_x[i, -1] < self.particle_p[i, -1]: self.particle_p[i] = self.particle_x[i]
    #
    def particle_swarm_optimization(self):
        minindex = np.argmin(self.particle_p[:, -1])
        self.particle_v = self.particle_v + self.c1 * (self.particle_p - self.particle_x) + self.c2 * (self.particle_p[minindex] - self.particle_x)
        self.particle_v[self.particle_v > 1] = 1
        self.particle_v[self.particle_v < -1] = -1
        self.particle_x = self.particle_x + self.particle_v

    def get_optimization_para(self, data_len, i):
        min_index = np.argmin(self.particle_x[:,-1])
        error_rate = 2*self.particle_x[min_index,-1]/data_len
        print("error rate: ", error_rate, "iterate time: ", i, "min index: ", min_index)
        best_gene = self.particle_x[min_index]
        return best_gene