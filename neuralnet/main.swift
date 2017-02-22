//
//  main.swift
//  neuralnet
//
//  Created by Tony Francis on 2/21/17.
//  Copyright © 2017 Tony Francis. All rights reserved.
//

import Foundation

var mnist = MNIST()
var trainingData = mnist.loadData(imagePath: "/Users/tonyfrancis/workspace/11364/neuralnet/neuralnet/train-labels-idx1-ubyte.data", labelPath: "/Users/tonyfrancis/workspace/11364/neuralnet/neuralnet/train-labels-idx1-ubyte.data")
var testData = mnist.loadData(imagePath: "/Users/tonyfrancis/workspace/11364/neuralnet/neuralnet/t10k-images-idx3-ubyte.data", labelPath: "/Users/tonyfrancis/workspace/11364/neuralnet/neuralnet/t10k-labels-idx1-ubyte.data")

var net = NeuralNetwork(inputSize: 784, outputSize: 10)
net.addLayer(layer: Layer(inputSize: 784, size: 30, actFn: Sigmoid()))
net.addLayer(layer: Layer(inputSize: 30, size: 10, actFn: Sigmoid()))

net.SGD(trainingData: trainingData, epochs: 30, miniBatchSize: 10, eta: 3.0, testData: testData)

