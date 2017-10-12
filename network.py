import numpy as np

class Neuron:
    ''' Data:
            SIGNAL that it recieved during forward pass
            OUTPUT that it calclates in the forward pass
            Delta that is calculated in the backward pass
        Behavior:
            it can calcute the OUTPUT when given the SIGNAL
    '''

    def __init__(self):
        self.data={"signal":0.0,"delta":0.0,"output":0.0}

    def activate(self):
        self.data["output"]=np.tanh(self.data["signal"])



class Network:
    '''
        Data:
            Layes - Each layer has a certain no of Neurons
                    Each layer is a list of neurons

            Weight matrices - matrices of Weight connecting neurons in the layers
        Behaviours:
            Forward pass- Takes the input and calculate signals for the the neurons of the next Layes
            Activate the layer and calculate the outupts
            Calulate the error at the output layer and backprapgate it
            Alter the weight according to the delta and the output between which the weigh is placed
    '''


    #Initiallizing network
    #no_of_nuerons_per_layer is a tuple that holds number of neurons(excluding bias units) in the layers. Tuple should be in a order (input_layer,hidden layer 1,hidden layer 2....,output layer)
    def __init__(self,no_of_nuerons_per_layer):
        #Dictionary of layers
        #Each layer is a list of neurons
        #keys are to be named as "input" for input layer, "output" for output layer, "hidden_1","hidden_2" etc for hidden layes
        self.layers={}

        #Layers iteration order
        #Important!
        #layers_order is a list containing the keys of the dictionary to be iterated in order
        #This is needed beacuse, dictionary iteraion doesnot happen in order
        self.layer_order=[]

        #Dictionary of weight martices
        #Keys goes same as layers
        self.weight_matrices={}

        #creating layers
        for counter,i in enumerate(no_of_nuerons_per_layer):
            if(counter==0):
                _layer=[]
                #Creating Neurons and adding to list
                #i+1 for the bias unit
                for j in range(0,i+1):
                    _layer.append(Neuron())

                #Adding input_layer to layes Dictionary
                self.layers["input"]=_layer

                #Adding the key to iteration sequence
                self.layer_order.append("input")

            elif(counter+1<len(no_of_nuerons_per_layer)):
                _layer=[]
                for j in range(0,i+1):
                    _layer.append(Neuron())

                self.layers["hidden_{}".format(counter)]=_layer
                self.layer_order.append("hidden_{}".format(counter))


            elif(counter+1==len(no_of_nuerons_per_layer)):
                _layer=[]
                #No i+1 because output layer is definit
                for j in range(0,i):
                    _layer.append(Neuron())

                self.layers["output"]=_layer
                self.layer_order.append("output")

        #Initiallizing weight matrices to random valus
        previous_key=""
        for key in self.layer_order:
            #Input layer dont has an weight matrix
            if(key!="input" and key!="output"):
                self.weight_matrices[key]=np.random.rand(len(self.layers[previous_key]),len(self.layers[key])-1)

            #output layer doesnot has the bais unit so no need of the -1
            elif(key=="output"):
                self.weight_matrices[key]=np.random.rand(len(self.layers[previous_key]),len(self.layers[key]))

            previous_key=key


        #Initialize outputs of bias units to 1
        for key,value in self.layers.items():
            if(key!="output"):
                value[0].data["output"]=1.0


        #Feedforward
        #Input vector is a (x+1,1) dimentional vector, where x is the size that was given when the network was instanciated
        def feed_forward(self,input_vector):
            for key in self.layer_order:
                signal_vectors=[]
                if(key!="input"):
                    _s=np.vdot(self.weight_matrices[key].transpose(),input_vector)








if(__name__=="__main__"):
    net=Network((256,20,10))

    # #Test for weight matrices
    # print("Testing weight matrices...")
    # print(net.weight_matrices["hidden_1"].shape)
    # print(net.weight_matrices["output"].shape)
    # print(net.layer_order)

    print("Testing value of bias units...")
    print(net.layers["hidden_1"][0].data["output"])
    print(net.layers["hidden_1"][1].data["output"])
