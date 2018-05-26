import os
import numpy as np
import math
from RBFN import pos

class RBFN(object):
    def __init__(self, populationSize, c1, c2, convergenceCondition, hiddenLayerNeuralNumber, trainData):
        self.particleswarmoptimization = pos.ParticleSwarmOptimization(populationSize, c1, c2, hiddenLayerNeuralNumber)
        self.hiddenLayerNeuralNumber = hiddenLayerNeuralNumber
        self.trainData = self._read_File(trainData)
        self.save_weight = open("weight.txt", 'w')
        self._train(convergenceCondition)
    #return steeringWheel to simulated environment
    def get_steeringWheel(self, straight, right, left):
        value = np.array([straight/2, right/2, left/2])
        value = (value - 40) / 40
        print(straight, right, left, value)
        sum = 0
        try:
            for i in range(self.hiddenLayerNeuralNumber+1):
                if i == 0:
                    sum += self.weight[0]
                else:
                    phi = np.exp(\
                        -(value - self.weight[1 + self.hiddenLayerNeuralNumber + (i-1)*3 : 1 + self.hiddenLayerNeuralNumber + (i-1)*3 + 3]) ** 2 / (2 * (self.weight[1 + self.hiddenLayerNeuralNumber + self.hiddenLayerNeuralNumber*3 + (i-1): 1 + self.hiddenLayerNeuralNumber + self.hiddenLayerNeuralNumber*3 + (i-1) + 1])**2) )
                    phi = np.sum(phi, axis=0)
                    sum += self.weight[i] * phi
            if math.isnan(sum):
                sum = 0
        except Exception as e:
            print(e)
            quit()
        sum = sum*80   #unnormalize
        print(sum)
        return sum
    #start to train RBFN by genetic_algorithm
    def _train(self, convergenceCondition):
        self.trainData = self._normalize_Train_Data(self.trainData)
        self.particleswarmoptimization.create_particle_x_p_v(self.trainData)
        for i in range(convergenceCondition):
            self.particleswarmoptimization.particle_swarm_optimization()
            self.particleswarmoptimization.calculate_Fitness_Function(self.trainData)
            self.particleswarmoptimization.chose_best_fitness_particle()

            self.weight = self.particleswarmoptimization.get_optimization_para(self.trainData.shape[0], i)
        # self.genetic_Algorithm.calculate_Fitness_Function(self.trainData)
        # self.weight = self.genetic_Algorithm.get_optimization_para(self.trainData.shape[0], 0)
        self.save_weight.write(np.array2string(self.weight))
        print("Weight: ", self.weight)
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
            # data = []
            # for x in string:
            #     for i in x:
            #         if i != '':
            #             data.append(i)
            string = [x[0:4] for x in string if x != '']
            print(string)
            data = np.array(string, dtype=np.float)
        except Exception as e:
            print(e)
            quit()

        return data

