maze = [
    [[1,0,1,1],[1,1,0,0],[1,0,1,1],[1,1,0,0]],
    [[1,0,0,1],[0,0,1,0],[1,0,0,0],[0,1,0,0]],
    [[0,1,1,1],[1,0,1,1],[0,1,1,0],[0,0,1,1]]
]

pIn = (0,0)
pOut = (2,4)
episode = 10

penalty = -1
reward = 10

qTab = {}

for x in range(0,len(maze)):
    for y in range(0,len(maze[0])):
        qTab[(x,y)] = 0
qTab[pOut] = reward

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

import random
def bot(pos, sPer = 0.125):
    moves = findPath(pos)
    if random.random() < sPer:
        return random.choice(moves)
    else:
        maxQ = float('-inf')
        for move in moves:
            qVal = qTab[move]
            if qVal > maxQ:
                maxQ = qVal
                bestMove = move
                updateQ(bestMove, pos)
        return bestMove

def updateQ(bMove, pMove):
    qTab[pMove] += 0.5*(penalty + qTab[bMove] * 0.9 - qTab[pMove])

def main():
    i=0
    while (i < episode):
        pos = pIn
        while pos != pOut:
            pos = bot(pos)
            print(pos)
        i+=1
        print("Episode: ", i)
        print("Final Run: ")
        pos = pIn
        while pos != pOut:
            pos = bot(pos, 0)
            print(pos)
        print("Q-Table: ", qTab)

main()