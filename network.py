#third-party libraries
import numpy as np

class Network(object): #network class
    def __init__(self, sizes):
        self.sizes = sizes #size
        self.numLayer = len(sizes)
        self.biases = [np.random.randn(r, 1) for r in sizes[1:]]
        self.weights = [np.random.randn(r, c) \
                        for r, c in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a): #neuron affects next layer
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, trainingData, epoch, miniBatchSize, eta, testData=None):
        n = len(trainingData)
        for e in range(epoch): #start an epoch of training
            np.random.shuffle(trainingData)
            miniBatches = [trainingData[k:k+miniBatchSize] \
                           for k in range(0, n, miniBatchSize)[:-1]]
            for batch in miniBatches:
                self.updateMiniBatch(batch, eta)

            if not testData == None:
                correct = self.evaluate(testData)
                print("Epoch {0}: \t{1} / {2}\taccuracy: {3:.2f}%".format( \
                    e, correct, len(testData), correct/len(testData)*100))
            else:
                print("Epoch {0} completed.".format(e))

    def updateMiniBatch(self, batch, eta): #gradient descend on miniBatch
        nablaB = [np.zeros(b.shape) for b in self.biases] #initialize gradient
        nablaW = [np.zeros(w.shape) for w in self.weights]

        for image, label in batch: #backpropagation, update B and W
            deltaNablaB, deltaNablaW = self.backprop(image, label) #backpropagation
            nablaB = [nB + dnB for nB, dnB in zip(nablaB, deltaNablaB)] #update nablaB
            nablaW = [nW + dnW for nW, dnW in zip(nablaW, deltaNablaW)] #update nablaW
        
        self.biases = [b-(eta/len(batch))*nB \
                       for b, nB in zip(self.biases, nablaB)] #update biases
        self.weights = [w-(eta/len(batch))*nW \
                        for w, nW in zip(self.weights, nablaW)] #update weights

    def backprop(self, image, label):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        active = image #first layer (input)
        activeLayer = [image] #stores all activation layers
        zs = [] #z vector list

        for b, w in zip(self.biases, self.weights): #feedforward
            z = np.dot(w, active)+b
            zs.append(z)
            active = self.sigmoid(z)
            activeLayer.append(active)

        delta = self.costDerivative(activeLayer[-1], label) * self.sigmoidPrime(zs[-1])
        nablaB[-1] = delta
        nablaW[-1] = np.dot(delta, activeLayer[-2].transpose())

        for l in range(2, self.numLayer):
            z = zs[-l]
            sp = self.sigmoidPrime(z)
            delta = np.dot(self.weights[1-l].transpose(), delta)*sp
            nablaB[-l] = delta
            nablaW[-l] = np.dot(delta, activeLayer[-1-l].transpose())
        return (nablaB, nablaW)

    def evaluate(self, testData):
        testResults = [(np.argmax(self.feedforward(image)), np.argmax(label)) \
                       for (image, label) in testData]
        return sum(int(guess == label) for (guess, label) in testResults)

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoidPrime(self, z):
        return self.sigmoid(z)*(1.0-self.sigmoid(z))

    def costDerivative(self, outputLayer, label):
        return (outputLayer - label)








    
