import mnistLoader
import network

trainingData = mnistLoader.loadTrainingData()
testData = mnistLoader.loadTestData()

net = network.Network([784, 16, 16, 10])
net.SGD(trainingData, 300, 100, 1, testData=trainingData)
