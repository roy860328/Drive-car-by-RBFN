import numpy as np

class Genetic_Algorithm(object):
    def __init__(self, populationSize, matingRate, mutationRate, hiddenLayerNeuralNumber, reproduceWay):
        self.reproduceWay = reproduceWay
        self.populationSize = populationSize
        self.matingRate = matingRate
        self.mutationRate = mutationRate
        self.hiddenLayerNeuralNumber = hiddenLayerNeuralNumber

        self._create_gene_pool()
    #Create gene pool with population size
    def _create_gene_pool(self):
        self.Dim = 1 + self.hiddenLayerNeuralNumber + 3*self.hiddenLayerNeuralNumber + self.hiddenLayerNeuralNumber
        self.gene_pool = [np.random.uniform(-1.0, 1.0, size=(self.Dim+1,)) for _ in range(int(self.populationSize))]
    #
    def _fitness_Function(self, trainData):

        return 0.5
    #
    def calculate_Fitness_Function(self, trainData):
        for i in range(len(self.gene_pool)):
            self.gene_pool[i][self.Dim] = self._fitness_Function(trainData)
    #
    def reproduce(self):
        if self.reproduceWay == "Competition":
            pass
        elif self.reproduceWay == "Turntable":
            pass
        pass
    #
    def mate(self):
        pass
    #
    def mutata(self):
        pass
    #
    def get_optimization_para(self):
        return 0
