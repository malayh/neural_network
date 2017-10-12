import numpy as np

'''Reads data from the data.txt file  in Data folder to make dataSet.npz in the same directory'''

def loadData(fileName):
    temp=[y.strip().split(" ") for y in [x for x in open("./Data/data.txt","r").read().strip().split("\n")]]
    dataSet=[[float(y) for y in x] for x in temp]


    data={}
    inputs=[]
    outputs=[]
    for entry in dataSet:
        inputs.append(np.array([entry[:256]]).transpose())
        outputs.append(np.array([entry[256:]]).transpose())

    data["inputs"]=inputs
    data["outputs"]=outputs

    return data


if(__name__=='__main__'):
    print("Loading data")
    dataSet=loadData("data.txt")
    print("Data Loaded")


    # np.savetxt("t.txt",dataSet["inputs"][0],newline="||",delimiter="\n\n")
    # np.savetxt("t.txt",dataSet["outputs"][0],newline="||",delimiter="\n\n")

    np.savez("./Data/dataSetNUMPY",dataSet["inputs"],dataSet["outputs"])
