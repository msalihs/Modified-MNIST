import numpy
import scipy.io as sio
from sklearn.model_selection import train_test_split
from scipy import optimize

class myNN:

	def __init__(self, hiddenLayerSize, alpha, weights = 0, lam = 0):
		self.hiddenLayerSize = hiddenLayerSize
		self.alpha = alpha
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
		
		if type(self.hiddenLayerSize) == int:
			self.numberOfHiddenlayers = 1
		else:
			self.numberOfHiddenlayers = len(self.hiddenLayerSize)

		if self.weights == 0:
			self.weights = self.initWeights()
			print "Weights matrix initialized with {a} and {b}".format(a = str(self.weights[0].shape), b = str(self.weights[1].shape))
		else:
			print "Weights matrix already given with {a} and {b}".format(a = str(self.weights[0].shape), b = str(self.weights[1].shape))

		#print "Weight matrix is set to {}".format(self.weights)
		print "Alpha is set to {}".format(self.alpha)
		print "Hidden layer is set to {}".format(self.hiddenLayerSize)
		print "Num of hidden layers is {}".format(self.numberOfHiddenlayers)
		print "Num of output is {}".format(self.outputLayerSize)

		cost = self.computeCost(x,y)
		dGrad = 1
		iteration = 0
		oldGrad = 0
		#res1 = optimize.fmin_cg(self.computeCost, self.weights, fprime=backPropogate)

		print "Cost of output is {}".format(cost)

		while (iteration < 1000):
			grad = self.backPropogate(x,y)
			self.weights = self.UpdateWeights(x,grad)
			cost = self.computeCost(x,y)
			iteration = iteration + 1
			dGrad = numpy.abs(oldGrad-numpy.sum(grad[0]))
			oldGrad = numpy.sum(grad[0])
			if iteration % 5 == 0:
				print "At {it}th \t iteration: Cost is {c}, D-Grad is {d}".format(it = iteration, c = cost, d = dGrad)

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

			epsilon = 0.12 #numpy.sqrt(6.0 / (layerIn + layerOut))
			w.append(2*epsilon*numpy.random.random((layerOut,layerIn)) - epsilon)
		return w

	def computeCost(self,x,y):
		layers = self.numberOfHiddenlayers + 1
		m = self.numberOfObservations
		l = self.outputLayerSize 
		w = self.weights
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
			yI[y[i] - 1] = 1.0

			#cost = cost + (0.5)*numpy.sum(numpy.square(yI - cOutput))
			cost = cost - numpy.sum((yI*numpy.log(cOutput)) + ((1-yI)*numpy.log(1-cOutput)))
			#for k in range(l):
			#	o = cOutput[k]				
			#	cost = cost - (yI[k]*numpy.log(o)) - ((1-yI[k])*numpy.log(1-o))

		regCost = 0
		for i in range(layers):
		 	regCost = regCost + numpy.sum(numpy.sum(numpy.square(w[i]))) - numpy.sum(numpy.sum(numpy.square(w[i][:,0]))) 
		return (1.0/m)*cost + (self.lam/(2.0*m))*regCost

	def backPropogate(self,x,y):
		layers = self.numberOfHiddenlayers + 1
		m = self.numberOfObservations
		l = self.outputLayerSize
		w = self.weights
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
			yI[y[i] - 1] = 1.0

			#print "processing {}".format(x[i])
			#print "propagate yielded {}".format(cOutputs)
			#print "y is {}".format(yI)

			# Compute Gradients going backwards 
			delta = []
			for j in reversed(range(layers)):
				sG = cOutputs[j]*(1-cOutputs[j])
				if (j+1) == layers:
					delta.append(sG * (yI - cOutputs[j]))
				else:
					pW = numpy.dot(w[j+1].T,delta[layers-j-2])
					delta.append(numpy.dot(sG,pW[1:]))

			#print "deltas yielded {}".format(delta)

			# Update the gradient
			for j in range(len(w)):
				if i == 0:
					grad.append(numpy.array(numpy.transpose(numpy.matrix(delta[len(w) - j - 1]))*numpy.matrix(cInputs[j])))					
				else:
					grad[j] = grad[j] + numpy.array(numpy.transpose(numpy.matrix(delta[len(w) - j - 1]))*numpy.matrix(cInputs[j]))

			#print "gradient yielded {}".format(grad)

			#raw_input("Press Enter to continue...")

		for i in range(len(grad)):
			regTerm = numpy.zeros(grad[i].shape)
			regTerm[:,1:] = grad[i][:,1:]
			grad[i] = (1.0/m)*grad[i] + (self.lam / m)*regTerm

		return grad

	def UpdateWeights(self, x, grad):
		layers = self.numberOfHiddenlayers + 1
		m = self.numberOfObservations
		w = self.weights
		for i in range(m):
			for j in range(layers):
				w[j] += grad[j]*self.alpha
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
			y.append(str(numpy.argmax(yP)+1))

		return y


def debug():
	dataContent = sio.loadmat('ex4data1.mat')
	weights = sio.loadmat('ex4weights.mat')
	x = dataContent['X']
	y = dataContent['y']
	th = []
	th.append(weights['Theta1'])
	th.append(weights['Theta2'])

	xTrain, xCV, yTrain, yCV = train_test_split(x, y, test_size=0.20, random_state=42)

	clf = myNN((25), 0.0005, lam = 1)

	#xTrain = [[1,1,1], [1,1,0],[0,0,1],[0,1,1],[0,0,0]]
	#yTrain = [1,1,0,1,1]

	w = clf.train(xTrain,yTrain)

	yP = clf.predict(xCV)

	return w, xCV, yCV, yP
