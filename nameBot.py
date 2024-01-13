from functools import reduce
import numpy as np
import random
from time import sleep
from string import ascii_lowercase as L
#random.seed(0)
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
    return (a-Y)
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
        He = lambda n: np.sqrt(2/n)
        for layer,dim in enumerate(topology):
            w1 = []
            b1 = []
            if layer == 0:
                for _ in range(dim):
                    tw=[]
                    b1.append(random.gauss(0,0.1))
                    for _ in range(inputDim):
                        tw.append(random.gauss(0,He(inputDim)))
                    w1.append(tw)
            else:
                #if layer == len(topology)-1: continue
                for _ in range(dim):
                    tw=[]
                    b1.append(random.gauss(0,0.1))
                    for _ in range(topology[layer-1]):
                        tw.append(random.gauss(0,He(topology[layer-1])))
                    w1.append(tw)
            self.W.append(w1)
            self.B.append(b1)
        else:#go again for weights between last hidden layer and output layer
            w1 = []
            b1 = []
            b1.append(random.gauss(0,0.1))
            for _ in range(dim):
                w1.append(random.gauss(0,He(topology[-1])))
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
    def cost(self, Y,a):
        return MSE(Y,a)
    def dCost(self, Y,a):
        return dMSE(Y, a)
    def backPropagation(self,Y,X,eta):
        #n = len(X)   
        #C=0
        #self.forwardProp(X)
        '''
        d0 = 0
        for i in range(n):
            self.forwardProp(X[i])
            C += self.cost(Y[i],self.output)/n
            d0 += self.dCost(Y[i],self.output)*self.output*(1-self.output)/n
        '''
        d = [[self.dCost(Y,self.output)*self.output*(1-self.output)]]
        C=self.cost(Y,self.output)
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
                    #print(self.topology)
                    if layer < len(self.W): self.W[-layer][tLi][pLi] -= eta* d[layer-1][tLi] * self.N[-(layer)][pLi]
                    else: self.W[-layer][tLi][pLi] -= eta* d[layer-1][tLi]*X[pLi]
                self.B[-layer][tLi] -= eta*d[layer-1][tLi]
        #self.printWeights()
        #input('done')
        #self.forwardProp(X)
        #print(self.cost(Y, self.output)-C)
    def backProp(self,Y,X,eta):
        d0 = 0
        C=0
        n = len(X)
        for i in range(n):
            self.forwardProp(X[i])
            d0 += self.dCost(Y[i],self.output)*self.output*(1-self.output)
        d = [[d0/n]]
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
        for i in range(n):
            for layer, weightMatrix in enumerate(self.W[::-1],start=1):
                for tLi, wList in enumerate(weightMatrix):
                    for pLi, _ in enumerate(wList):
                        if layer < len(self.W): self.W[-layer][tLi][pLi] -= eta* d[layer-1][tLi] * ReLu(self.N[-(layer)][pLi])
                        else: self.W[-layer][tLi][pLi] -= eta* d[layer-1][tLi]*X[i][pLi]
                    self.B[-layer][tLi] -= eta*d[layer-1][tLi]
        #self.printWeights()
        #input('done')
        for x in X:
            self.forwardProp(x)
            C += self.cost(Y[i],self.output)/n
        print(C)
        sleep(.01)
    def dReLu(self, x):
        # Derivative of ReLU activation function
        return 1 if x > 0 else 0

    def backwardProp(self, IL, Y, lr):
        # Backward Propagation

        # Calculate error in the output layer
        self.error = self.cost(Y, self.output)
        
        # Calculate the gradient at the output layer
        dOutput = self.dCost(Y, self.output) * dSigmoid(self.output)
        
        # Update weights and biases in the output layer
        for i in range(self.topology[-1]):
            for j in range(len(self.N[-1])):
                self.W[-1][0][i] -= lr * dOutput * self.N[-1][j]
            self.B[-1][0] -= lr * dOutput

        # Backpropagate the error through hidden layers
        for layer in reversed(range(len(self.topology))):
            for k in range(self.topology[layer]):
                # Calculate the gradient at the hidden layer
                if layer == len(self.topology) - 1:
                    self.errors[layer][k] = dOutput * self.W[-1][0][k]
                else:
                    self.errors[layer][k] = self.errors[layer+1][0] * self.W[layer+1][0][k] * self.dReLu(self.N[layer][k])

                # Update weights and biases in the hidden layer
                for j in range(len(IL) if layer == 0 else self.topology[layer-1]):
                    self.W[layer][k][j] -= lr * self.errors[layer][k] * (IL[j] if layer == 0 else self.N[layer-1][j])
                self.B[layer][k] -= lr * self.errors[layer][k]

    def train(self, X, Y, lr, epochs):        
        avgLoss = -1
        for epoch in range(epochs):
            for i in range(len(X)):
                # Forward Pass
                self.forwardProp(X[i])

                # Backward Propagation
                self.backPropagation(Y[i],X[i], lr)

            # Print the loss for every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Average Loss: {round(avgLoss,10)}')
                avgLoss = 0
            avgLoss += self.cost(Y[i],self.output)/100

# Example usage:
# X_train, Y_train are your training data
# inputDim is the dimension of your input
# topology is a list describing hidden layers
'''
NameBot = BCNN([3,4],10)
#print(NameBot.W)
print('pred:',NameBot.predict(ohe("Chuka")))
for i in range(10):    
    NameBot.backpropagation(1,ohe("Chuka"),.1)
print('pred:',NameBot.predict(ohe("Chuka")))

'''
XOR = BCNN([3,2],2)
pre = XOR.predict([1,1])
batchSize = 30
seen = 0
print()
FCN = lambda a,b: a^b
Xs = [[random.randint(0,1),random.randint(0,1)] for _ in range(1000)]
Ys = [FCN(x1,x0) for x0, x1 in Xs]
try:
    XOR.train(Xs,Ys,.07,201)
except KeyboardInterrupt:
    pass
possible = [[i,j] for j in range(2) for i in range(2)]
print(''.join([f"{x}, {round(XOR.predict(x),8)}, {round(XOR.predict(x))}\n" for x in possible]))
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
