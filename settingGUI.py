import os
import sys
import tkinter as tk
#
import numpy as np

import pygamegraphic
from RBFN import RBFN


class GUI():
    def __init__(self):
        self.setInterface()
    def setInterface(self):

        self.interface = tk.Tk()
        # 創造視窗
        self.interface.title('self.interface')
        self.interface.geometry('500x600')
        # Add object
        self.createLableEntry2Interface()
        # 列出txt檔案
        self.createSelectionFile()
        
        # 訓練按鈕
        trainbtn = tk.Button(self.interface, text="train", command=self.clickTrainBtn)
        trainbtn.pack()
        # outputresult
        outputresult = tk.Label(self.interface, text="outputresult")
        outputresult.pack()
        # input learnrate
        self.outputresultprint = tk.StringVar()
        outputresultLabel = tk.Label(self.interface, textvariable=self.outputresultprint)
        outputresultLabel.pack()

        # 讓視窗實現
        self.interface.mainloop()

    def createLableEntry2Interface(self):
        # population size
        populationSize = tk.Label(self.interface, text="Population Size (particle)")
        populationSize.pack()
        self.populationSizeInput = tk.Entry(self.interface)  # input Population Size
        self.populationSizeInput.insert(0, "400")
        self.populationSizeInput.pack()
        # acceleration constants c1
        c1Label = tk.Label(self.interface, text="Acceleration constants: c1")
        c1Label.pack()
        self.c1LabelInput = tk.Entry(self.interface)  # input Mating Rate
        self.c1LabelInput.insert(0, "0.6")
        self.c1LabelInput.pack()
        # acceleration constants c2
        c2Label = tk.Label(self.interface, text="Acceleration constants: c2")
        c2Label.pack()
        self.c2LabelInput = tk.Entry(self.interface)  # input Mutation Rate
        self.c2LabelInput.insert(0, "0.4")
        self.c2LabelInput.pack()
        # 收斂字幕
        convergence = tk.Label(self.interface, text="Convergence (train times)")
        convergence.pack()
        self.convergenceInput = tk.Entry(self.interface)  # input Convergence
        self.convergenceInput.insert(0, "100")
        self.convergenceInput.pack()
        # Hidden Layer Neural Number
        hiddenLayerNeuralNumber = tk.Label(self.interface, text="Hidden Layer Neural Number")
        hiddenLayerNeuralNumber.pack()
        self.hiddenLayerNeuralNumberInput = tk.Entry(self.interface)  # input Mutation Rate
        self.hiddenLayerNeuralNumberInput.insert(0, "2")
        self.hiddenLayerNeuralNumberInput.pack()
        #TrainLabel
        TrainLabel = tk.Label(self.interface, text="Train data")
        TrainLabel.pack()

    def createSelectionFile(self):
        ###################################Train data list#########################

        self.trainDataListTxt = tk.Listbox(self.interface)
        # os.path.dirname(sys.executable)當產出exe檔時才能正確找到txt檔案位置,但無法在.py檔中使用
        # os.getcwd()只有在.py檔有用,因為exe檔的默認位置在"cd ~" 讀檔時會找不到檔案
        # print("sys.executable directory: ", os.path.dirname(sys.executable))
        # dist = os.path.dirname(sys.executable).rsplit('/', 1)
        # dist = dist[0] + "/Dataset/Hopfield_dataset"
        dist = os.getcwd()
        dist = dist + "/traindata"
        print("\n\ndist: ", dist)
        haveTxt = ''
        for file in os.listdir(dist):
            if file.endswith(".txt") or file.endswith(".TXT"):
                haveTxt += str(file) + ','
        haveTxt = haveTxt.split(",")
        haveTxt = list(filter(None, haveTxt))
        for txt in haveTxt:
            self.trainDataListTxt.insert(0, txt)
        self.trainDataListTxt.pack()
        
    def clickTrainBtn(self):
        #get列表選取的txt檔案
        selectionTrainData = self.trainDataListTxt.curselection()
        selectionTrainData = self.trainDataListTxt.get(selectionTrainData)
        selectionTrainData = os.path.dirname(__file__) + "/traindata/" + selectionTrainData

        #get Population Size, Mating Rate, Mutation Rate, Convergence, Hidden Layer Neural Number
        populationSize = float(self.populationSizeInput.get())
        c1Label = float(self.c1LabelInput.get())
        c2Label = float(self.c2LabelInput.get())
        convergenceCondition = int(self.convergenceInput.get())
        hiddenLayerNeuralNumber = int(self.hiddenLayerNeuralNumberInput.get())

        self.RBFN = RBFN.RBFN(populationSize, c1Label, c2Label, convergenceCondition, \
                                hiddenLayerNeuralNumber, selectionTrainData)

        pygamegraphic.mainPygame(self.RBFN)