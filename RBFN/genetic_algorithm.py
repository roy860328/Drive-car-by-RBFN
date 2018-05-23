import numpy as np

class Genetic_Algorithm(object):
    def __init__(self, populationSize, matingRate, mutationRate, hiddenLayerNeuralNumber, reproduceWay):
        self.reproduceWay = reproduceWay
        self.populationSize = int(populationSize)
        self.matingRate = matingRate
        self.mutationRate = mutationRate
        self.hiddenLayerNeuralNumber = hiddenLayerNeuralNumber

        self._create_gene_pool()
    #Create gene pool with population size
    #(theta, w1 ,w2 , …, wj , m11, m12, …, m1i, m21, m22, …, m2i, …, mj1, mj2,…, mji, σ1, σ2, …, σj)
    def _create_gene_pool(self):
        self.Dim = 1 + self.hiddenLayerNeuralNumber + 3*self.hiddenLayerNeuralNumber + self.hiddenLayerNeuralNumber
        self.gene_pool = [np.random.uniform(-1.0, 1.0, size=(self.Dim+1,)) for _ in range(int(self.populationSize))]
        self.gene_pool = np.array(self.gene_pool)
    #
    def _fitness_Function(self, gene_pool, trainData):
        sum = 0
        for i in range(self.hiddenLayerNeuralNumber+1):
            if i == 0:
                sum += gene_pool[0]
            else:
                sum += gene_pool[i] * self._gaussian_basis_function(trainData[:,:3], gene_pool[1 + self.hiddenLayerNeuralNumber + (i-1)*3 : 1 + self.hiddenLayerNeuralNumber + (i-1)*3 + 3], \
                                                     gene_pool[1 + self.hiddenLayerNeuralNumber + self.hiddenLayerNeuralNumber*3 + (i-1): 1 + self.hiddenLayerNeuralNumber + self.hiddenLayerNeuralNumber*3 + (i-1) + 1])
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
        for i in range(len(self.gene_pool)):
            self.gene_pool[i][self.Dim] = self._fitness_Function(self.gene_pool[i], trainData)
    #
    def reproduce(self):
        if self.reproduceWay == "Competition":
            self.gene_pool = self.gene_pool[self.gene_pool[:,-1].argsort()]
            if self.gene_pool.shape[0]%2 == 0:
                self.gene_pool[int(self.gene_pool.shape[0]/2):, :] = self.gene_pool[
                                                                       0:int(self.gene_pool.shape[0]/2), :]
            else:
                self.gene_pool[int(self.gene_pool.shape[0]/2):, :] = self.gene_pool[
                                                                     0:int(self.gene_pool.shape[0] / 2)+1, :]
        elif self.reproduceWay == "Turntable":
            self.gene_pool = self.gene_pool[self.gene_pool[:, -1].argsort()]
            probability = np.array(self.gene_pool[:,-1])
            print(np.max(probability), np.min(probability))
            probability = np.max(probability)-probability
            probability = (probability / np.sum(probability))
            select_list = np.random.choice(self.populationSize, self.populationSize, p=probability)
            # print(select_list)
            select_list = np.sort(select_list)
            self.gene_pool = self.gene_pool[select_list]
            self.gene_pool = self.gene_pool[self.gene_pool[:, -1].argsort()]
            # probability = np.array(self.gene_pool[:, -1])
            # print(np.max(probability), np.min(probability))

    #
    def mate(self):
        radom_index = np.random.choice(self.populationSize, self.populationSize, replace=False)
        self.gene_pool = self.gene_pool[radom_index]
        for i in range(len(self.gene_pool)-1):
            if np.random.rand(1)[0] < self.matingRate and i%2 == 0:
                # print(i)
                temp = np.array(self.gene_pool[i])
                self.gene_pool[i] = self.gene_pool[i] + np.random.rand(1)[0]*(temp - self.gene_pool[i+1])
                self.gene_pool[i+1] = self.gene_pool[i+1] - np.random.rand(1)[0]*(temp - self.gene_pool[i+1])
    #
    def mutate(self):
        for i in range(len(self.gene_pool) - 1):
            if np.random.rand(1)[0] < self.mutationRate:
                self.gene_pool[i] = self.gene_pool[i] + np.random.rand(1)[0] * (np.random.uniform(-1.0, 1.0, size=(self.Dim+1,)))
    #
    def get_optimization_para(self, data_len, i):
        min_index = np.argmin(self.gene_pool[:,-1])
        error_rate = 2*self.gene_pool[min_index,-1]/data_len
        print("error rate: ", error_rate, "iterate time: ", i)
        best_gene = self.gene_pool[min_index]
        return best_gene
