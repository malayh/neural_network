import numpy as np
from random import shuffle

class Layer:
    #Every layer is has a dictionary
    #that dictionary contians three vectors - signal,delta,output
    def __init__(self):
        #signal_vector ,delta_vector and output_vector will later have numpy vectors
        self.data={"signal_vector":0,"delta_vector":0,"output_vector":0}

    def activate(self):
        self.data["output_vector"]=np.tanh(self.data["signal_vector"])



class Network:
    #A Network is a list of layers

    #no_of_nuerons_per_layer is a tuple says how many neurons are needed in a single layer
    def __init__(self,no_of_nuerons_per_layer):
        #list of layers
        self.layers=[]

        #network has weight matrices mapping each layer to the next
        #weight_matrices is a list of numpy matrices
        self.weight_matrices=[]

        #creating layers
        for i in no_of_nuerons_per_layer:
            self.layers.append(Layer())

        #Initiallizing the weight matrices
        for count,num in enumerate(no_of_nuerons_per_layer):
            if(count!=0):
                _wm=np.random.rand(no_of_nuerons_per_layer[count-1]+1,num)
                self.weight_matrices.append(_wm)


    #forward pass
    #form pass calculates signal_vector, activates layers
    #value of input_vector is same as the no of nuerons in the 1st layer as mentioned when network was instanciated
    def feed_forward(self,input_vector):
        #preparing the input valus by adding the bias unit's value that is 1
        _temp=[1]
        for i in input_vector:
            _temp.append(i[0])

        bais_added_iv=np.array([_temp]).transpose()

        #Assigne the bais_added_iv to the output of the 1st layer
        self.layers[0].data["output_vector"]=bais_added_iv

        #forwar pass the the output of the 1st layer to calculate other layers
        for count,layer in enumerate(self.layers):
            #excluding inpute layer and output layer
            if(count!=0 and count!=len(self.layers)-1):
                layer.data["signal_vector"]=np.dot(self.weight_matrices[count-1].transpose(),self.layers[count-1].data["output_vector"])
                layer.activate()
                #Adding bais value to the output_vector (not for output layer)
                _temp=[1]
                for i in layer.data["output_vector"]:
                    _temp.append(i[0])

                layer.data["output_vector"]=np.array([_temp]).transpose()
            #for the ouput layer
            elif(count==len(self.layers)-1):
                layer.data["signal_vector"]=np.dot(self.weight_matrices[count-1].transpose(),self.layers[count-1].data["output_vector"])
                layer.activate()



    #Squared error, just to display progress
    def error(self,expected,output):
        _temp=expected-output
        _temp=_temp**2
        return np.sum(_temp)



    #backwar pass
    #output vetor is the expected result for the input vector
    #feed forward a input vector and calulate error at output layer
    #Backpropagate the error and calcuate the delta_vectors
    #Alter weights according to the formula
    def backward_pass(self,input_vector,output_vector):
        #feed the input vector forward
        self.feed_forward(input_vector)

        #calculating delta_vector foe the output layer
        self.layers[len(self.layers)-1].data["delta_vector"]=2*(self.layers[len(self.layers)-1].data["output_vector"]-output_vector)*(1-self.layers[len(self.layers)-1].data["output_vector"]**2)

        #calculating delta by backpropagating delta of the output layer
        for count,layer in reversed(list(enumerate(self.layers))):
            if(count!=0):
                delta_j=layer.data["delta_vector"]
                w_i_j=self.weight_matrices[count-1]
                x_i=self.layers[count-1].data["output_vector"]

                delta_i=(1-x_i**2)*(np.dot(w_i_j,delta_j))

                #Removing the delta calulate for the biased unit
                _temp=[]
                for c,i in enumerate(delta_i):
                    if(c!=0):
                        _temp.append(i[0])

                delta_i=np.array([_temp]).transpose()

                self.layers[count-1].data["delta_vector"]=delta_i

        #calculating of the gradient matrices that are to be substracted from the weight_matrices
        #weight_gradient is a list of matrices,each matrix is of same size as of the matrices in weight_matrices according to their indices
        weight_gradient=[]
        for count,layer in enumerate(self.layers):
            if(count!=0):
                _gradient=np.outer(self.layers[count-1].data["output_vector"],layer.data["delta_vector"])
                weight_gradient.append(_gradient)

        return weight_gradient


    #THIS IS A TEST train method
    # def train(self,input_vector,output_vector):
    #     while(self.error(output_vector,self.layers[len(self.layers)-1].data["output_vector"])>0.00001):
    #         wg=self.backward_pass(input_vector,output_vector)
    #         for count,wm in enumerate(self.weight_matrices):
    #             self.weight_matrices[count]-=0.1*wg[count]
    #         print(self.error(output_vector,self.layers[len(self.layers)-1].data["output_vector"]))




    #train method will train the Network
    #dataSet will acccept a list of dictionaries, each dictionary having input and output pairs, indexed as "input","output"
    #it will acccept learning rate as an real number, which at default is 0.1
    #stop_at specify the error at which algorithm stop iterating

    def train(self,dataSet,learning_rate=0.1,stop_at=0.01):
        it=0
        while(1):
            count_it=0
            error=0

            shuffle(dataSet)

            for dataPoint in dataSet:
                weight_gradient=self.backward_pass(dataPoint["input"],dataPoint["output"])

                #changing the weights using the weight_gradient calculated
                for count,weights in enumerate(self.weight_matrices):
                    self.weight_matrices[count]-=learning_rate*weight_gradient[count]

                it+=1
                error+=self.error(dataPoint["output"],self.layers[len(self.layers)-1].data["output_vector"])

            if(count_it==5000):
                count_it=0
                learning_rate=learning_rate-0.00001

            #After one epoch
            print("Iterations:{}    Error:{}".format(it,error/len(dataSet)))




            if(error/len(dataSet)<stop_at):
                #Write weights to file and exit
                np.savez("./Data/weight_matrices",self.weight_matrices)
                break

    def predict(self,input_vector,thrashold=0.9):
        self.feed_forward(input_vector)
        _output=self.layers[len(self.layers)-1].data["output_vector"]
        _temp=[]
        for i in _output:
            if(i[0]>=thrashold):
                _temp.append(1)
            else:
                _temp.append(0)

        return np.array([_temp]).transpose()



    def load_weights(self,weight_matrices):
        self.weight_matrices=weight_matrices



if(__name__=="__main__"):
    net=Network((256,5,3,10))


    npz=np.load("./Data/trainingSet.npz")
    dataSet=[]
    for i,o in zip(npz["arr_0"],npz["arr_1"]):
        _dict={}
        _dict["input"]=i
        _dict["output"]=o

        dataSet.append(_dict)
    net.train(dataSet,learning_rate=0.015,stop_at=0.905)
