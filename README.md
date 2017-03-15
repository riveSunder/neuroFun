    Basic functions for training neural networks and RBMs based on my notes
    from Geoffrey Hinton's Coursera course
	 (https://www.coursera.org/learn/neural-networks)

Function List, neural networks:
	sigmoid(z)
	sigmoidGradient(z)
	forProp(myInputs,myWeights)
	backProp(myInputs,myCrossInputs,myTarget,myWeights,myIter=10,lR=1e-5,myLambda=0,myMom=1e-5,dropout=False): 
	
Function List RBMs:
	hidToVis(rbmW,hidStates)
	visToHid(rbmW,visStates)
	sampPRand(myInput,seed=1)
	myGoodness(rbmW,hidStates,visStates)
	myGoodnessGradient(hidStates,visStates)
	trainRBMLayers(a0,hiddenLayers,lR,myIter)

