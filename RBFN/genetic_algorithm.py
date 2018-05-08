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
                sum += gene_pool[i] * self._gaussian_basis_function(trainData[:,:3], gene_pool[self.hiddenLayerNeuralNumber + i*3 : self.hiddenLayerNeuralNumber + i*3 + 3], \
                                                     gene_pool[self.hiddenLayerNeuralNumber + self.hiddenLayerNeuralNumber*3 + i: self.hiddenLayerNeuralNumber + self.hiddenLayerNeuralNumber*3 + i + 1])
        fitness = (trainData[:,3]-sum)**2
        fitnesssum = np.sum(fitness, axis = 0)

        return fitnesssum
    #trainData=[none, 3], m=[1, 3], sigma[1, 3]
    def _gaussian_basis_function(self, trainData, m, sigma):
        phi = np.exp(-(trainData - m)**2/(2*sigma))
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
                self.gene_pool[int(self.gene_pool.shape[0]/2) :, :] = self.gene_pool[0: int(self.gene_pool.shape[0]/2), :]
            else:
                self.gene_pool[int(self.gene_pool.shape[0] / 2):, :] = self.gene_pool[
                                                                       0: int(self.gene_pool.shape[0] / 2)+1, :]
        elif self.reproduceWay == "Turntable":
            probability = self.gene_pool[:,-1]
            probability = probability / np.sum(probability)
            select_list = np.random.choice(self.populationSize, self.populationSize, p=probability)
            self.gene_pool = self.gene_pool[select_list]
            pass
        pass
    #
    def mate(self):
        pass
    #
    def mutate(self):
        pass
    #
    def get_optimization_para(self):
        return 0
