from functools import reduce
import numpy as np
import random
from string import ascii_lowercase as L
random.seed(0)
def ohe(name):
    #11 is max
    nameVec=[]
    tempList=[]
    for i in range(len(name)-1):
        tempList.append((name[i].lower(),name[i+1]))
    for pair in tempList:
        try:
            nameVec.append((L.index(pair[0])*len(L)+L.index(pair[1]))/(len(L)**2-1))
        except ValueError:
            input(pair)
    return nameVec + [0]*(10-len(nameVec))
def CE(Y,a):
    return Y*np.log(a)+(1-Y)*np.log(1-a)
def dCE(Y,a):
    return (Y/a) + (1-Y)/(1-a)
def MSE(Y,a):
    return (Y-a)**2
def dMSE(Y,a):
    return (Y-a)
def ReLu(x):
    return np.maximum(0,x)
def dReLu(x):
    return x>=0
def sigmoid(x):
    return 1/(1+np.exp(-x))
def dSigmoid(x):
    z = sigmoid(x)
    return z*(1-z)
def invSigmoid(x):
    return np.log(x/(1-x))
class BCNN:
    #asserting input of length 10
    def __init__(self, topology, inputDim):
        self.W = []
        self.B = []
        self.N = [[0] * dim for dim in topology]
        self.topology = topology # describes hidden layers
        self.errors = [[0.0] * dim for dim in self.topology]
        self.output = 0
        self.error = 0
        for layer,dim in enumerate(topology):
            w1 = []
            b1 = []
            if layer == 0:
                for _ in range(dim):
                    tw=[]
                    b1.append(random.random())
                    for _ in range(inputDim):
                        tw.append(random.uniform(-1,1))
                    w1.append(tw)
            else:
                #if layer == len(topology)-1: continue
                for _ in range(dim):
                    tw=[]
                    b1.append(random.random())
                    for _ in range(topology[layer-1]):
                        tw.append(random.uniform(-1,1))
                    w1.append(tw)
            self.W.append(w1)
            self.B.append(b1)
        else:#go again for weights between last hidden layer and output layer
            w1 = []
            b1 = []
            b1.append(random.random())
            for _ in range(dim):
                w1.append(random.uniform(-1,1))
            self.W.append([w1])
            self.B.append(b1)
    def setZero(self):
        self.output = 0
        self.N = [[0] * dim for dim in self.topology]
    def forwardProp(self,IL):
        self.setZero()
        #j is current layer, k is next layer
        for layer,dim in enumerate(self.topology):
            if layer == 0:
                for k in range(dim):
                    for j in range(len(IL)):
                        self.N[layer][k] += self.W[layer][k][j]*IL[j]
                    self.N[layer][k] += self.B[layer][k]
                    self.N[layer][k] = ReLu(self.N[layer][k])
            else:                 
                for k in range(dim):
                    for j in range(self.topology[layer-1]):
                        self.N[layer][k] += self.W[layer][k][j]*self.N[layer-1][j]
                    self.N[layer][k] += self.B[layer][k]
                    self.N[layer][k] = ReLu(self.N[layer][k])
        
        for i in range(self.topology[-1]):
            self.output+=self.N[-1][i]*self.W[-1][0][i]
        self.output+=self.B[-1][0]
        self.output = sigmoid(self.output)
        
        #self.print()
    def predict(self, inp):
        self.forwardProp(inp)
        return self.output
    def print(self):
        neur = 0
        for layer in self.N:
            print(layer)
            print()
    def printWeights(self):
        for i in self.W:
            print(i)
        print()
    def backPropagation(self,Y,X,eta):
        self.forwardProp(X)
        C = CE(Y,self.output)
        d = [[dCE(Y,self.output)*self.output*(1-self.output)]]
        shape = [1]+self.topology[::-1]
        #print(shape)
        
        for layer,dim in enumerate(shape[1:],start=1):
            dL=[0]*dim
            for j in range(dim):
                for k in range(shape[layer-1]):
                    dL[j] += d[-1][k]*self.W[-layer][k][j] * dReLu(self.N[-layer][j])
            d.append(dL)
        #input(d[-1])
        for layer, weightMatrix in enumerate(self.W[::-1],start=1):
            pass#print(layer)
        #input()
        for layer, weightMatrix in enumerate(self.W[::-1],start=1):
            for tLi, wList in enumerate(weightMatrix):
                for pLi, _ in enumerate(wList):
                    if layer < len(self.W): self.W[-layer][tLi][pLi] -= eta* d[layer-1][tLi] * self.N[-(layer)][pLi]
                    else: self.W[-layer][tLi][pLi] -= eta* d[layer-1][tLi]*X[pLi]
                self.B[-layer][tLi] -= eta*d[layer-1][tLi]
        #self.printWeights()
        #input('done')
        self.forwardProp(X)
        print(CE(Y,self.output)-C)
        '''

    def computeError(self):
        self.errors = [[0] * dim for dim in self.topology]
        num_of_layers = len(self.topology)
        
        for i in range(self.topology[num_of_layers - 1]):
            # assuming ReLU as activation function for derivative
            self.errors[num_of_layers - 1][i] = self.error * self.W[-1][0][i] * dReLu(self.N[-1][i])
        
        # computing error for other hidden layer neurons
        for layer in range(num_of_layers - 2, -1, -1):
            for i in range(self.topology[layer]):
                error = 0.0
                for j in range(self.topology[layer+1]):
                    error += self.errors[layer+1][j] * self.W[layer+1][j][i]
                    self.errors[layer][i] = error * dReLu(self.N[layer][i])
    def backProp(self, Y, X, eta):
        self.forwardProp(X)
        self.error = dMSE(Y, self.output)*self.output*(1-self.output)
        self.computeError()
        for layer, dim in enumerate(self.topology[::-1],start=1):
            wMatrix = self.W[-layer]
            for nextLayer, weightList in enumerate(wMatrix):
                for prevLayer, weight in enumerate(weightList):
                    self.W[-layer][nextLayer][prevLayer] -= eta* self.errors
        print(self.N)
        print(self.errors)
    '''
'''
NameBot = BCNN([3,4],10)
#print(NameBot.W)
print('pred:',NameBot.predict(ohe("Chuka")))
for i in range(10):    
    NameBot.backpropagation(1,ohe("Chuka"),.1)
print('pred:',NameBot.predict(ohe("Chuka")))

'''
XOR = BCNN([2,3],2)
print(XOR.predict([1,1]))
print()
for i in range(100):
    x0,x1 = random.randint(0,1),random.randint(0,1)
    XOR.backPropagation(x0^x1,[x0,x1],.05)
print()
print(XOR.predict([0,0]),XOR.predict([0,1]))
input()

def oneHot(Y):
    oneHotY = np.zeros((Y.size,2))
    for i, _ in enumerate(oneHotY):
        oneHotY[i][int(Y[i])] = 1
    #oneHotY[np.arange(Y.size),Y] = 1
    return oneHotY.T
with open("names_female.txt") as w:
    femaleNames = map(lambda x: x.replace('\n',''),w.readlines())
    w.close()
with open("names_male.txt") as m:
    maleNames = map(lambda x: x.replace('\n',''),m.readlines())
    m.close()

#totalLen = lambda x, y: len(x)+len(y) if type(x)==str else x+len(y)
#femLen = list(map(len,femaleNames))
#mascLen = list(map(len,maleNames))
X = [[0]+ohe(name) for name in femaleNames]+[[1]+ohe(name) for name in maleNames if len(name) < 11]
#print(ohe("Chuka"))
input()
#print(yTrain)
#input(xTrain[:,0])
