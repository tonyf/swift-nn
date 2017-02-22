//
//  NeuralNetwork.swift
//  neuralnet
//
//  Created by Tony Francis on 2/21/17.
//  Copyright © 2017 Tony Francis. All rights reserved.
//

import Foundation

typealias Data = [(input: [Double], target: [Double])]
struct DataList {
    var data: Data
    let num: UInt
}

class NeuralNetwork {
    private var num_layers = 2;
    var inputSize: UInt
    var layers:[Layer] = []
    
    init(inputSize: UInt, outputSize: UInt)
    {
        self.inputSize = inputSize
    }

    func addLayer(layer: Layer) -> Void
    {
        self.layers.append(layer)
    }
    
    func feedForward(input: [Double]) -> [Double]
    {
        var output: ([Double], UInt) = (input, self.inputSize)
        for layer in self.layers {
            output = layer.compute(input: output.0, inputSize: output.1)
        }
        return output.0
    }
    
    func evaluate(testData: DataList) -> Int
    {
        let outputs: [[Double]] = testData.data.map{ elem in self.feedForward(input: elem.input) }
        var num = 0
        for (i, output) in outputs.enumerated() {
            num += output == testData.data[i].target ? 1 : 0
        }
        return num
    }
    
    func SGD(trainingData: DataList, epochs: Int, miniBatchSize: Int, eta: Double, testData: DataList?) {
        for e in 1...epochs {
            var data = trainingData.data.shuffled()
            let miniBatches = stride(from: 0, to: data.count, by: miniBatchSize).map {
                Array(data[$0..<min($0 + miniBatchSize, data.count)])
            }
            for batch in miniBatches {
                self.updateMiniBatch(miniBatch: batch, eta: eta)
            }
            if (testData != nil) {
                print("Epoch \(e): \(self.evaluate(testData: testData!)) / \(testData!.num)")
            } else {
                print("Epoch \(e) complete.")
            }
        }
    }
    
    func updateMiniBatch(miniBatch:Data, eta:Double)
    {
        for elem in miniBatch {
            self.backprop(input: elem.input, target: elem.target)
        }
        
        for layer in layers {
            layer.updateWeights(miniBatchSize: miniBatch.count, eta: eta)
        }
    }
    
    func backprop(input: [Double], target: [Double])
    {
        let output = feedForward(input: input)
        let cost = Quad().Delta(output: output, target: target)
        var delta = layers.last!.initError(cost: cost)
        
        for i in stride(from: layers.count-2, to: 0, by: -1) {
            delta = layers[i].updateError(delta: delta, nextLayer: layers[i+1])
        }
    }
}

// Array Shuffle
// Source: http://stackoverflow.com/a/24029847
// Could use built in shuffle function but this is only supported on macOS

extension MutableCollection where Indices.Iterator.Element == Index {
    /// Shuffles the contents of this collection.
    mutating func shuffle() {
        let c = count
        guard c > 1 else { return }
        
        for (firstUnshuffled , unshuffledCount) in zip(indices, stride(from: c, to: 1, by: -1)) {
            let d: IndexDistance = numericCast(arc4random_uniform(numericCast(unshuffledCount)))
            guard d != 0 else { continue }
            let i = index(firstUnshuffled, offsetBy: d)
            swap(&self[firstUnshuffled], &self[i])
        }
    }
}

extension Sequence {
    /// Returns an array with the contents of this sequence, shuffled.
    func shuffled() -> [Iterator.Element] {
        var result = Array(self)
        result.shuffle()
        return result
    }
}
