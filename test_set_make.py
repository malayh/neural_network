import numpy as np
from random import randint

npz=np.load("./Data/dataSet.npz")

#counting how many examples for each digit
count=[0 for x in npz["arr_1"][0]]

for i in npz["arr_1"]:
    for c,j in enumerate(i):
        if(j[0]==1):
            count[c]+=1


dataSet={}
dataSet["inputs"]=list(npz["arr_0"])
dataSet["outputs"]=list(npz["arr_1"])

testSet={}
inputs=[]
outputs=[]


#picking random data points for test data set
def whatDigit(npa):
    #Returns the digit an output vector identifies
    for c,i in enumerate(npa):
        if(i[0]==1):
            return c

counter_for_digits=[0 for x in range(0,10)]
while(1):
    randIndex=randint(0,len(dataSet["outputs"]))
    wd=whatDigit(dataSet["outputs"][randIndex])
    if(counter_for_digits[wd]<15):
        counter_for_digits[wd]+=1
        inputs.append(dataSet["inputs"][randIndex])
        outputs.append(dataSet["outputs"][randIndex])

        del dataSet["outputs"][randIndex]
        del dataSet["inputs"][randIndex]

    if(sum(counter_for_digits)==150):
        break


#Teting test set
def printVector(npv):
    c=0
    for i in range(0,16):
        for j in range(0,16):
            if(npv[c,0]==1):
                print("##",end='')
            elif(npv[c,0]==0):
                print("   ",end='')

            c+=1

        print("\n")

    print("\n\n")


for c,i in enumerate(inputs):
    printVector(inputs[c])
    print(whatDigit(outputs[c]))


#Saving trainig set and test set to npz files
np.savez("./Data/trainigSet",dataSet["inputs"],dataSet["outputs"])
np.savez("./Data/testSet",inputs,outputs)
