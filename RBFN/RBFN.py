import os
import numpy as np
from RBFN import genetic_algorithm

class RBFN(object):
    def __init__(self, populationSize, matingRate, mutationRate, convergenceCondition, hiddenLayerNeuralNumber, reproduceWay, trainData):
        self.genetic_Algorithm = genetic_algorithm.Genetic_Algorithm(populationSize, matingRate, mutationRate, hiddenLayerNeuralNumber, reproduceWay)
        self.hiddenLayerNeuralNumber = hiddenLayerNeuralNumber
        self.trainData = self._read_File(trainData)

        self._train(convergenceCondition)
    #return steeringWheel to simulated environment
    def get_SteeringWheel(straight, right, left):
        steeringWheel = 0
        return steeringWheel
    #start to train RBFN by genetic_algorithm
    def _train(self, convergenceCondition):
        self.trainData = self._normalize_Train_Data(self.trainData)
        for i in range(convergenceCondition):
            self.genetic_Algorithm.calculate_Fitness_Function(self.trainData)
            self.genetic_Algorithm.reproduce()
            self.genetic_Algorithm.mate()
            self.genetic_Algorithm.mutate()
        self.w = self.genetic_Algorithm.get_optimization_para()
    #Normalize train data
    def _normalize_Train_Data(self, trainData):
        trainData[:,3] = trainData[:,3]/40
        trainData[:,0:3] = (trainData[:,0:3]-40)/40
        return trainData
    #Read train data
    def _read_File(self, file):
        try:
            string = ""
            pfile1 = open(file, "r")
            string = pfile1.read()
            string = string.split('\n')
            # string to double list
            string = [i.split(' ') for i in string]
            string = [x for x in string if x != ['']]
            strin = string
            x = 1
        except Exception as e:
            print(e)

        return np.array(string, dtype=float)

