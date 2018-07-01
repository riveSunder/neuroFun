### neuroFun.py
### Basic functions for training neural networks and restricted 
### Boltzmann machines
### www.github.com/thescinder/neuroFun
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    #Returns the logistic of the value z
    mySig = 1 / (1+np.exp(-z))
    return mySig

def sigmoidGradient(z):
    #return the gradient of a sigmoid function at value z
    mySigGrad = sigmoid(z)*(1-sigmoid(z))
    return mySigGrad

def reLU(z):
    myReLU = np.max([[np.zeros(len(x))],[x]],axis=0)	
    return myReLU

def reLUGradient(z):
	myReLUGrad = np.zeros(len(z))
	for c in range(0,z):
		if (z[c] > 0):
			myReLUGrad[c] = 1.0
		else:
			myReLUGrad[c] = 0.0
	return myReLUGrad 

def softplus(z):
    mySoftplus = np.log(1+np.exp(z))
    return mySoftplus
   
def softplusGradient(z):
    mySoftplusGrad = sigmoid(z)
    return(mySoftplusGrad)

def hidToVis(rbmW,hidStates):
    visProb = np.dot(rbmW.T,hidStates)
    visProb = sigmoid(visProb)
    return visProb

def visToHid(rbmW,visStates):
    hidProb = np.dot(rbmW,visStates)
    hidProb = sigmoid(hidProb)
    return hidProb

def initRBMW(hLayer,vLayer,mySeed=1.0):
    np.random.seed(mySeed)
    rbmW = np.random.random((hLayer,vLayer))
    return rbmW

def sampPRand(myInput,seed=1):
    #Compare input to pseudo-random variables
    myTest = myInput > np.random.random(np.shape(myInput))
    return myTest * 1

def myGoodness(rbmW,hidStates,visStates):
    #m = np.shape(visStates)[1]
    E = - np.mean(np.dot(np.dot(rbmW, visStates).T, hidStates));
    G = -(E); 
    return G
    
def myGoodnessGrad(hidStates,visStates):
    m = np.shape(visStates)[1];
    myGG = np.dot(visStates,hidStates.T)
    myGG = myGG.T/ m;
    return myGG

def trainRBMLayers(a0,hiddenLayers,lR,myIter):
    #Train an RBM layer based on visible input layer a0
    #a0 - visible units
    #hiddenLayers - number of hidden layers
    #lR - learning rate
    #myIter - number of iterations to train
    myTest = a0
    print('debug')
    J = []
    print('test')
    rbmW = initRBMW(hiddenLayers,myTest.shape[0],1)
    for j in range(myIter):
        myTest = sampPRand(a0)
        myHid = visToHid(rbmW,myTest)
        myHid0 = myHid
        #print(myHid)
        myHid = sampPRand(myHid)
        #print(myHid[:,10])
        myDream = hidToVis(rbmW,myHid)
        E = (a0-myDream)
        J.append(np.mean(np.abs(E)))
        myDream = sampPRand(myDream)
        myReconProb = visToHid(rbmW,myDream)
        myRecon = sampPRand(myReconProb)
        myPos = myGoodnessGrad(myHid,myTest)

        myNeg = myGoodnessGrad(myRecon,myDream)
        rbmW = rbmW + lR* (myPos-myNeg)
        if ( j % (myIter/10) == 0):
            G = myGoodness(rbmW,myHid,myTest)
            print("Iteration " + str(j)+" Error = " + str(np.mean(np.abs(E))))
            print("Goodness = " + str(G))
            
    plt.plot(J)
    plt.show()
    print("Finished with RBM training of size " + str(np.shape(rbmW.T)))
    return rbmW, myReconProb

#forward propagation 
def forProp(myInputs,myWeights,dORate=0.0):
    #print(np.shape(myInputs))
    #print(np.shape(myWeights[0]))
    zn = [] #np.dot(myInputs,myWeights[0])
    an = [] #sigmoid(zn)
    myDOWeights = []

    if(0):
        for n in range(len(myWeights)):
            zn.append(np.zeros((      np.shape(myWeights[n][1])[0],np.shape(a0)[0])))
            an.append(np.zeros((      np.shape(myWeights[n][1])[0],np.shape(a0)[0])))
        
    
    #zn[0,:] = np.array([np.dot(myInputs,myWeights[0])])
    #an[0,:] = sigmoid(zn[0,:])
    if(0):
        zn[0] = np.dot(myInputs.T,myWeights[0])
        an[0] = sigmoid(zn[0])
    np.random.seed(1)
	
    if(dORate):
        dO = np.random.random(np.shape(myWeights[0]))
        dO[dO>dORate] = 1.0
        dO[dO<1.0] = 0.0
        myDOWeights.append(dO*myWeights[0])
        zn.append(np.squeeze(np.dot(myInputs.T,myDOWeights[0])))
    else:	
        zn.append(np.squeeze(np.dot(myInputs.T,myWeights[0])))
        #print(np.shape(zn))
    an.append(sigmoid(zn[0]))
    #print(np.shape(zn[0]))
    for n in range(1,len(myWeights)):
        #print(np.shape(an))
        #print(np.shape(myWeights[n]))
        if(0):
            print(n)
            zn[n] = np.dot(an[n-1],myWeights[n])
            an[n] = sigmoid(zn[n])

        if(dORate):
            dO = np.random.random(np.shape(myWeights[n]))
            dO[dO>dORate] = 1.0
            dO[dO<1.0] = 0.0
            myDOWeights.append(dO*myWeights[n])
            zn.append(np.squeeze(np.dot(an[n-1],myDOWeights[n])))
        else:	
            zn.append(np.dot(an[n-1],myWeights[n]))
        an.append(sigmoid(zn[n]))
        #print(np.shape(an[n]))
        #print(np.shape(an))
    
    return an,zn, myDOWeights



#back propagation function

def backProp(myInputs,myCrossInputs,myTarget,myWeights,myIter=10,lR=1e-5,myLambda=0,myMom=1e-5,stopEarly=False,dORate=0.0):
    #init momentum
    momSpeed = []
    #init weight penalties
    wPen = []
    #init gradients
    dGrad = []
    Delta = []
    J = []
    if(stopEarly):
        bestE = 2
    myBestWeights = np.copy(myWeights)
    for n in range(len(myWeights)): #-1,-1,-1):
        wPen.append(0*myWeights[n])
        dGrad.append(0*myWeights[n])
        momSpeed.append(0*myWeights[n])
        Delta.append(0*myWeights[n])
        #print(np.shape(wPen[n]))
        #print(np.shape(dGrad[n]))
        #print(np.shape(momSpeed[n]))
    m = m = np.shape(myInputs)[1]
    myFreq = int(myIter/10)
  
    print("Begin Training . . . ")
    for i in range(myIter):
        #Run forward propagation.
        myOutput, myZ, myDOWeights = forProp(myInputs,myWeights,dORate)
        #print(np.shape(myOutput))
        #print(np.shape(myZ[0]))
        #use squared error as objective function
        E = (myTarget.T-myOutput[len(myOutput)-1])
        if(stopEarly):
            # Remember the best weights so far
            myE = np.mean(np.abs(E))
            #print(myE)
            if(myE < bestE):
                bestE = myE
                myBestWeights = myWeights
        J.append(np.mean(np.abs(E)))
        d = []
        #print(np.shape(E.T))
        d.append(E.T)
        #print(np.shape(d[0]))
        #d.append(E.T)
       

        if(i%myFreq == 0):
            print("Iteration " + str(i) + " Mean error = "+str(np.mean(np.abs(E))))


        for n in range(len(myWeights)-1,-1,-1):
            if(dORate):
                d.append(np.dot(myDOWeights[n],d[len(d)-1]) * sigmoidGradient(myZ[n-1].T))
            else:
            	d.append(np.dot(myWeights[n],d[len(d)-1]) * sigmoidGradient(myZ[n-1].T))
           

        for n in (range(len(myWeights)-1,-1,-1)):
           
            for i in range(m-1):
                #
                Delta[n] =Delta[n] + np.dot(np.array([myOutput[n-1][i,:]]).T,np.array([d[len(d)-(n+2)][:,i]]))
               
        
        for n in (range(len(myWeights)-1,-1,-1)):
            wPen[n] = myLambda * myWeights[n]

        for n in (range(len(myWeights)-1,-1,-1)):
            #print(n)
            dGrad[n] = Delta[n]/m +wPen[n]
            momSpeed[n] = myMom*momSpeed[n] + dGrad[n]
            
            myWeights[n] = myWeights[n] + momSpeed[n] * lR
            #print(np.mean(Theta3))
        #print("Training Finished, avg error = "+str(np.mean(np.abs(E))))
        #print(E)
    if (stopEarly):
        print(bestE)
    #   myWeights = myBestWeights
    #Return current weights, best weights, and a history of the cost function
    return myWeights, myBestWeights, J 

