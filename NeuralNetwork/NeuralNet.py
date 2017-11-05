import numpy
import sklearn
import scipy.io as sio
from sklearn.model_selection import train_test_split
from scipy import optimize

class myNN:

	def __init__(self, hiddenLayerSize, alpha, weights = 0, lam = 0, beta = 0):
		self.hiddenLayerSize = hiddenLayerSize
		self.alpha = alpha
		self.beta = beta
		self.lam = lam
		if type(weights) == list:
			self.weights = list(weights)
		else:
			self.weights = weights


	def train(self,x,y):
		if len(x) != len(y):
			raise NameError('Dimension Mismatch')

		self.inputLayerSize = len(x[0])
		self.numberOfObservations = len(x)
		self.outputLayerSize = len(numpy.unique(y))
		self.xTrain = x
		self.yTrain = y % 10
		self.w_d = []
		self.epoch = 0
		# self.yTrain = numpy.zeros(y.shape)
		# for i in range(len(y)):
		# 	self.yTrain[i] = y[i] % 10
		
		if type(self.hiddenLayerSize) == int:
			self.numberOfHiddenlayers = 1
		else:
			self.numberOfHiddenlayers = len(self.hiddenLayerSize)

		if self.weights == 0:
			self.weights = self.initWeights()
			print "Weights matrix initialized with {a} and {b}".format(a = str(self.weights[0].shape), b = str(self.weights[1].shape))
		else:
			print "Weights matrix already given with {a} and {b}".format(a = str(self.weights[0].shape), b = str(self.weights[1].shape))
			self.w_d = self.weights[:]
			for i in range(len(w_d)):
				w_d[i] = numpy.zeros(w_d[i].shape)

		#print "Weight matrix is set to {}".format(self.weights)
		print "Alpha is set to {}".format(self.alpha)
		print "Hidden layer is set to {}".format(self.hiddenLayerSize)
		print "Num of hidden layers is {}".format(self.numberOfHiddenlayers)
		print "Num of output is {}".format(self.outputLayerSize)

		w_r = self.roll(self.weights)
		dGrad = 1
		oldGrad = 0
		#w_opt = optimize.fmin_ncg(self.computeCost, w_r, fprime=self.backPropogate)
		#self.weights = self.unRoll(w_opt)
		#cost = self.computeCost(w_opt)
		#print "Cost of output is {}".format(cost)

		while (dGrad > 0.0001) and (self.epoch < 100):
			grad = self.backPropogate(w_r)
			self.weights = self.UpdateWeights(x,grad)
			w_r = self.roll(self.weights)
			cost = self.computeCost(w_r)
			self.epoch = self.epoch + 1
			dGrad = numpy.sum(numpy.abs(grad))
			if self.epoch % 5 == 0:
				print "At {it}th \t iteration: Cost is {c}, D-Grad is {d}".format(it = self.epoch, c = cost, d = dGrad)

		return self.weights


	def initWeights(self):
		nInput = self.inputLayerSize
		nOutput = self.outputLayerSize
		nHidden = []
		w = []

		if type(self.hiddenLayerSize) == int:
			nHidden.append(self.hiddenLayerSize)
		else:
			nHidden = self.hiddenLayerSize

		size = self.numberOfHiddenlayers + 1
		for i in range(size):
			if i == 0:
				layerIn = nInput + 1
			else:
				layerIn = nHidden[i-1] + 1

			if (i + 1) == size:
				layerOut = nOutput
			else:
				layerOut = nHidden[i]

			epsilon = numpy.sqrt(6.0 / (layerIn + layerOut))
			w.append(2*epsilon*numpy.random.random((layerOut,layerIn)) - epsilon)
			self.w_d.append(numpy.zeros(w[i].shape))
		return w

	def computeCost(self,w_r):
		layers = self.numberOfHiddenlayers + 1
		m = self.numberOfObservations
		l = self.outputLayerSize 
		x = self.xTrain
		y = self.yTrain
		w = self.unRoll(w_r)
		cost = 0
		for i in range(m):
			for j in range(layers):
				if j == 0:
					cInput = numpy.ones(len(x[i])+1)
					cInput[1:] = x[i]
				else:
					cInput = numpy.ones(len(cOutput)+1)
					cInput[1:] = cOutput
				cOutput = self.sigmoid(numpy.dot(w[j],cInput))
			# yI is the ideal y output
			yI = numpy.zeros(l)
			yI[y[i]] = 1.0

			cost = cost - numpy.sum((yI*numpy.log(cOutput)) + ((1-yI)*numpy.log(1-cOutput)))

		regCost = 0
		for i in range(layers):
		 	regCost = regCost + numpy.sum(numpy.sum(numpy.square(w[i]))) - numpy.sum(numpy.sum(numpy.square(w[i][:,0]))) 
		return (1.0/m)*cost + (self.lam/(2.0*m))*regCost

	def backPropogate(self,w_r):
		layers = self.numberOfHiddenlayers + 1
		m = self.numberOfObservations
		l = self.outputLayerSize
		x = self.xTrain
		y = self.yTrain
		w = self.unRoll(w_r)
		grad = []
		# Go through every sample
		for i in range(m):
			cOutputs = []
			cInputs  = []
			for j in range(layers):
				if j == 0:
					cInput = numpy.ones(len(x[i])+1)
					cInput[1:] = x[i]
				else:
					cInput = numpy.ones(len(cOutput)+1)
					cInput[1:] = cOutput
				cOutput = self.sigmoid(numpy.dot(w[j],cInput))
				cInputs.append(cInput)
				cOutputs.append(cOutput)
			yI = numpy.zeros(l)
			yI[y[i]] = 1.0

			#print "processing {}".format(x[i])
			#print "propagate yielded {}".format(cOutputs)
			#print "y is {}".format(yI)

			# Compute Gradients going backwards 
			delta = []
			for j in reversed(range(layers)):
				sG = cOutputs[j]*(1-cOutputs[j])
				if (j+1) == layers:
					temp = sG * (yI - cOutputs[j])
				else:
					pW = numpy.dot(w[j+1].T,delta[layers-j-2])
					temp = sG*pW[1:]
				delta.append(temp)

			delta.reverse()

			# Update the gradient
			for j in range(len(w)):
				if i == 0:
					grad.append(numpy.array(numpy.matrix(delta[j]).T*numpy.matrix(cInputs[j])))
				else:
					grad[j] = grad[j] + numpy.array(numpy.matrix(delta[j]).T*numpy.matrix(cInputs[j]))

			#print "gradient yielded {}".format(grad)

		for i in range(len(grad)):
			regTerm = numpy.zeros(grad[i].shape)
			regTerm[:,1:] = grad[i][:,1:]
			grad[i] = (1.0/m)*grad[i] + (self.lam / m)*regTerm

		grad_r = self.roll(grad)

		return grad_r

	def UpdateWeights(self, x, grad_r):
		layers = self.numberOfHiddenlayers + 1
		m = self.numberOfObservations
		w = self.weights
		w_d = self.w_d
		grad = self.unRoll(grad_r)
		for i in range(m):
			for j in range(layers):
				w_d[j] = grad[j]*self.alpha + w_d[j]*self.beta
				w[j] += w_d[j]
		return w

	def sigmoid(self,x):
		return 1. / (1 + numpy.exp(-x))

	def sigmoidGrad(self,x):
		#g = self.sigmoid(x)
		return numpy.dot(g, (1-g))

	def softmax(self, x):
		e_x = numpy.exp(x - numpy.max(x))
		return e_x / e_x.sum()

	def predict(self,x):
		if self.weights == 0:
			raise NameError('DidNotTrainYet')
		m = len(x)
		layers = self.numberOfHiddenlayers + 1
		w = self.weights
		y = []
		for i in range(m):
			for j in range(layers):
				if j == 0:
					cInput = numpy.ones(len(x[i])+1)
					cInput[1:] = x[i]
				else:
					cInput = numpy.ones(len(cOutput)+1)
					cInput[1:] = cOutput
				cOutput = self.sigmoid(numpy.dot(w[j],cInput))
			# yI is the ideal y output
			yP = self.softmax(cOutput)
			y.append(str(numpy.argmax(yP)))

		return y

	def roll(self,w):
		for i in range(self.numberOfHiddenlayers+1):
			temp = numpy.array(w[i]).ravel()
			if i == 0:
				rolledW = temp
			else:
				rolledW = numpy.concatenate((rolledW,temp))
		return rolledW

	def unRoll(self,w):
		unRolledW = []
		index = 0
		nInput = self.inputLayerSize
		nOutput = self.outputLayerSize
		size = self.numberOfHiddenlayers + 1
		nHidden = []

		if type(self.hiddenLayerSize) == int:
			nHidden.append(self.hiddenLayerSize)
		else:
			nHidden = self.hiddenLayerSize

		for i in range(size):
			if i == 0:
				layerIn = nInput + 1
			else:
				layerIn = nHidden[i-1] + 1

			if (i + 1) == size:
				layerOut = nOutput
			else:
				layerOut = nHidden[i]
			unRolledW.append(numpy.reshape(w[index:(index+layerIn*layerOut)],(layerOut,layerIn)))
			index = index+layerIn*layerOut
		return unRolledW

def debug():
	dataContent = sio.loadmat('ex4data1.mat')
	weights = sio.loadmat('ex4weights.mat')
	x = dataContent['X']
	y = dataContent['y']
	th = []
	th.append(weights['Theta1'])
	th.append(weights['Theta2'])

	xTrain, xCV, yTrain, yCV = train_test_split(x, y, test_size=0.20, random_state=42)

	clf = myNN((25), 5e-4, lam = 0.5, beta = 0.25)

	#xTrain = [[1,1,1], [1,1,0],[0,0,1],[0,1,1],[0,0,0]]
	#yTrain = [1,1,0,1,1]

	w = clf.train(xTrain,yTrain)

	yP = clf.predict(xCV)
	yCV = yCV % 10
	y1 = []
	y2 = []
	for i in range(len(yCV)):
	    y1.append(int(yP[i]))
	    y2.append(int(yCV[i]))

	accuracy = sklearn.metrics.accuracy_score(y1,y2)

	print "Accuracy is {}".format(accuracy)

	return w, xCV, y2, y1
