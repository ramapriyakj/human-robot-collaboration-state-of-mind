from StateOfMInd_TF import *
from random import *
import threading

def printit():
    threading.Timer(3.0,printit).start()
    states = []
    stateofstates = []
    statesstring = []
    stateofmind = ""
    y = randint(0, 4)
    for i in range(0,30):
        x = randint(0,y)
        if x == 0:
            statesstring.append("  SIT")
        elif x == 1:
            statesstring.append("STAND")
        elif x == 2:
            statesstring.append(" LOOK")
        elif x == 3:
            statesstring.append(" WARN")
        elif x == 4:
            statesstring.append("GRASP")
        states.append(int(x))

    stateofstates.append(states)
    result = getpred(stateofstates)

    if result == 0:
        stateofmind = "TIRED"
    elif result == 1:
        stateofmind = "DISTRACTED"
    elif result == 2:
        stateofmind = "WORKING"
    elif result == 3:
        stateofmind = "INCAPABLE"

    print("\nStates are :")
    matrix = [statesstring[p:p+5] for p in range(0,len(statesstring),5)]
    for l in matrix:
        print(l);
    print("\nState of Mind : "+stateofmind+"\n");

printit()




