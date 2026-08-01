maze = [
    [[1,0,1,1],[1,1,0,0],[1,0,1,1],[1,1,0,0]],
    [[1,0,0,1],[0,0,1,0],[1,0,0,0],[0,1,0,0]],
    [[0,1,1,1],[1,0,1,1],[0,1,1,0],[0,0,1,1]]
]

pIn = (0,0)
pOut = (2,4)
episode = 10
learningRate = 0.9

inputLayer = 8
hiddenLayer = 64
hLN = 2
outputLayer = 4

def indexOfMax(arr):
    for i in range(len(arr)):
        if(max(arr) == arr[i]):
            return i

def findPos(val, pos):
    if val == 0:
        return (pos[0]-1, pos[1])
    if val == 1:
        return (pos[0], pos[1]+1)
    if val == 2:
        return (pos[0]+1, pos[1])
    if val == 3:
        return (pos[0], pos[1]-1)


def findPath(pos):
    block = maze[pos[0]][pos[1]]
    movable = []
    if block[0] == 0:
        movable.append((pos[0]-1, pos[1]))
    if block[1] == 0:
        movable.append((pos[0], pos[1]+1))
    if block[2] == 0:
        movable.append((pos[0]+1, pos[1]))
    if block[3] == 0:
        movable.append((pos[0], pos[1]-1))
    return movable

def mul(a,b):
    sol = [[0 for j in range(len(b[0]))] for i in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            s=0
            for k in range(len(b)):
                s += a[i][k] * b[k][j]
            sol[i][j] = s
    return sol

def delA(y,a,w):
    return learningRate * 2*(y-a)*w

def delB(y,a):
    return learningRate * 2*(y-a)

def delW(y,a,x):
    return learningRate * (2*(y-a)*x)

def ReLu(x):
    if x < 0:
        return 0
    else:
        return x
    
import random
def deepQLearn(nn):
    pos = pIn   
    while (pos != pOut):

        if (random.random() < 0.125):
            ans = [random.choice(findPath(pos))]

        else:
            ans = nn.forward([[
                maze[pos[0]][pos[1]][0],
                maze[pos[0]][pos[1]][1],
                maze[pos[0]][pos[1]][2],
                maze[pos[0]][pos[1]][3],
                pos[0],
                pos[1],
                pOut[0],
                pOut[1]]])
        
        

        if(findPos(indexOfMax(ans),pos) in findPath(pos)):
            nn.backProp(nn.forward([[
            maze[pos[0]][pos[1]][0],
            maze[pos[0]][pos[1]][1],
            maze[pos[0]][pos[1]][2],
            maze[pos[0]][pos[1]][3],
            findPos(indexOfMax(ans),pos)[0],
            findPos(indexOfMax(ans),pos)[1],
            pOut[0],
            pOut[1]]]),hLN+2)
            pos=findPos(indexOfMax(ans),pos)
    
class NeuralNetwork:
    def __init__(self):

        self.lRbuffer = []

        self.weights = []
        self.biases = []
        m1 = [[0 for j in range(hiddenLayer)] for i in range(inputLayer)]
        self.weights.append(m1)

        bias=[]
        for b in range(hiddenLayer):
            bias.append(0)
        self.biases.append(bias)

        for k in range(hLN-1):
            m2 = [[0 for j in range(hiddenLayer)] for i in range(hiddenLayer)]
            self.weights.append(m2)
            bias=[]
            for b in range(hiddenLayer):
                bias.append(0)
            self.biases.append(bias)

        m3 = [[0 for j in range(outputLayer)] for i in range(hiddenLayer)]
        self.weights.append(m3)
        bias=[]
        for b in range(outputLayer):
            bias.append(0)
        self.biases.append(bias)

    def forward(self, input):
        self.lRbuffer.append(input)
        layerVal = mul(input, self.weights[0]) 
        for k in range(len(layerVal)):
            layerVal[0][k] = layerVal[0][k] + self.biases[0][k]
        for i in range(len(layerVal)):
            for j in range(len(layerVal[i])):
                layerVal[i][j] = ReLu(layerVal[i][j])

        self.lRbuffer.append(layerVal)

        for i in range(1, len(self.weights)):
            layerVal = mul(layerVal, self.weights[i])
            for l in range(len(layerVal)):
                layerVal[0][l] = layerVal[0][l] + self.biases[i][l]
            for j in range(len(layerVal)):
                for k in range(len(layerVal[j])):
                    layerVal[j][k] = ReLu(layerVal[j][k])

            self.lRbuffer.append(layerVal)

        return max(layerVal)
    
    def backProp(self, y, lN):
        if (lN == 0):
            return
        weights = self.weights[lN-1]
        bWeights = self.weights[lN-1]
        biases = self.biases[lN-1]
        layer = self.lRbuffer[lN]
        prevLayer = self.lRbuffer[lN-1]
        
        for i in range(len(layer)):
            biases[i] += delB(y[i], layer[0][i])
        
        for i in range(len(layer[0])):
            for j in range(len(weights)):
                weights[j][i] += delW(y[i], layer[0][i], prevLayer[0][j])
        
        for i in range(len(layer[0])):
            for j in range(len(prevLayer)):
                prevLayer[0][j] += delA(y[i], layer[0][i], bWeights[j][i]) 

        self.backProp(prevLayer[0],lN-1)      


deepQLearn(NeuralNetwork())