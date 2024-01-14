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

def CE(Y,Y_pred):
    epsilon = 1e-9
    return -Y*np.log(Y_pred + epsilon) + (1 - Y)*np.log(1 - Y_pred + epsilon)

def dCE(Y,Y_pred):
    epsilon = 1e-9
    return (-Y/(Y_pred + epsilon)) + (1 - Y)/(1 - Y_pred + epsilon)

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

def addMatrix(matrix1, matrix2):
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions for addition")
    result = [[0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            result[i][j] = matrix1[i][j] + matrix2[i][j]
    return result

def scaleMatrix(factor, matrix):
    result = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[i][j] = matrix[i][j] * factor
    return result

class BCNN:
    #asserting input of length 10
    def __init__(self, topology, inputDim):
        self.W = []
        self.B = []
        self.N = [[0] * dim for dim in topology]
        self.topology = topology # describes hidden layers
        self.output = 0
        
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
        self.mw = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.B]
        self.vw = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.B]
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
    def backPropSGD(self,Y,X,eta):
        d0 = 0
        dW = [scaleMatrix(0,w) for w in self.W]
        dB = [[0 for _ in range(len(dmi))] for dmi in self.B]
        shape = [1]+self.topology[::-1]
        #C=0
        n = len(X)
        for i in range(n):
            self.forwardProp(X[i])
            d0 += self.dCost(Y[i],self.output)*self.output*(1-self.output)
            d = [[d0]]
            #print(shape)
            
            for layer,dim in enumerate(shape[1:],start=1):
                dL=[0]*dim
                for j in range(dim):
                    for k in range(shape[layer-1]):
                        dL[j] += d[-1][k]*self.W[-layer][k][j] * dReLu(self.N[-layer][j])
                d.append(dL)
        
            for layer, weightMatrix in enumerate(self.W[::-1],start=1):
                for tLi, wList in enumerate(weightMatrix):
                    for pLi, _ in enumerate(wList):
                        if layer < len(self.W): dW[-layer][tLi][pLi] -= d[layer-1][tLi] * ReLu(self.N[-(layer)][pLi])/n
                        else: dW[-layer][tLi][pLi] -= d[layer-1][tLi]*X[i][pLi]/n
                    dB[-layer][tLi] -= d[layer-1][tLi]/n
        

        for i,bL in enumerate(self.B):
            for j,bias in enumerate(bL):
                self.B[i][j] = bias + eta*dB[i][j]
                
        #print(len(self.W),len(self.W[0]))
        #print(len(dW),len(self.B[0]))
        
        for i,x in enumerate(self.W):
            self.W[i] = addMatrix(x, scaleMatrix(eta,dW[i]))
    def backPropagation1(self, Y, X, eta, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        #TODO: Adam version of SGD
        pass
    
    def train(self, X, Y, lr, epochs, batchSize=1):        
        avgLoss = -1        
        possible = [[i,j] for j in range(2) for i in range(2)]
        for epoch in range(epochs):
            if batchSize==1:
                for i in range(0,len(X)):
                    # Forward Pass
                    self.forwardProp(X[i])

                    # Backward Propagation
                    self.backPropagation(Y[i],X[i], lr)
            else:
                for i in range(0,len(X),batchSize):
                    # Backward Propagation
                    self.backPropSGD(Y[i:i+batchSize],X[i:i+batchSize], lr)
            # Print the loss for every 100 epochs
            if epoch % 100 == 0:
                print(f'\nEpoch {epoch}, Average Loss: {round(avgLoss,10)}')
                avgLoss = 0
            avgLoss += self.cost(Y[i],self.output)/100
            print(round(avgLoss*(100/(1+epoch%100)),8),end='\r')



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
##XOR = BCNN([8],2)
##pre = XOR.predict([1,1])
##seen = 0
##print()
##FCN = lambda a,b: a^b
##Xs = [[random.randint(0,1),random.randint(0,1)] for _ in range(1000)]
##Ys = [FCN(x1,x0) for x0, x1 in Xs]
##try:
##    XOR.train(Xs,Ys,.05,301,10)
##except KeyboardInterrupt:
##    pass
##print('\n')
##possible = [[i,j] for j in range(2) for i in range(2)]
##print(''.join([f"{x}, {round(XOR.predict(x),8)}, {round(XOR.predict(x))}\n" for x in possible]))
##input()
#totalLen = lambda x, y: len(x)+len(y) if type(x)==str else x+len(y)
#femLen = list(map(len,femaleNames))
#mascLen = list(map(len,maleNames))
Z = [[0]+ohe(name) for name in femaleNames]+[[1]+ohe(name) for name in maleNames if len(name) < 11]
random.shuffle(Z)
Y = [z[0] for z in Z]
X = [z[1:] for z in Z]
#input(len(X[0]))
#print(ohe("Chuka"))
NameBot = BCNN([16,8], 10)
NameBot.train(X,Y,lr=.05,epochs=150,batchSize=50)
print(NameBot.predict(ohe("Grace")))
input(NameBot.predict(ohe("Josh")))
#print(yTrain)
#input(xTrain[:,0])
