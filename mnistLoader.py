#python standard libraries
import gzip
import struct

#third-party libraries
import numpy as np
import cv2

currDir = "./data/"
images = currDir + "train-images-idx3-ubyte.gz"
labels = currDir + "train-labels-idx1-ubyte.gz"

testImage = currDir + "t10k-images-idx3-ubyte.gz"
testLabel = currDir + "t10k-labels-idx1-ubyte.gz"

def readRawData(path):
    file = gzip.open(path, "rb")
    magic, num = struct.unpack(">II", file.read(8))

    if magic == 2049: #read label file
        label = struct.unpack(">"+"B"*num, file.read(num))
        label = [vectResult(n) for n in label]
        return label

    if magic == 2051: #read image file
        row, col = struct.unpack(">II", file.read(8))
        image = np.fromstring(file.read(), dtype=np.uint8)
        image = np.reshape(image, (num, row*col, 1))
        return image

def vectResult(n):
    v = np.zeros((10, 1))
    v[n] = 1.0
    return v


def loadTrainingData():
    trainingData = np.asarray(list(zip(readRawData(images),
                                       readRawData(labels))))
    return trainingData

def loadTestData():
    testData = np.asarray(list(zip(readRawData(testImage),
                                   readRawData(testLabel))))
    return testData
