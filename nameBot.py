import numpy as np
import random
from BCNN import BCNN
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
