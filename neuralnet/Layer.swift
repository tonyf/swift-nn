//
//  Layer.swift
//  neuralnet
//
//  Created by Tony Francis on 2/21/17.
//  Copyright © 2017 Tony Francis. All rights reserved.
//

import Foundation
import Accelerate

typealias ActivationFunction = (Double) -> Double

class Layer {
    var size: UInt
    var inputSize: UInt
    var biases: [Double]?
    var weights: [Double]?
    
    var d_biases: [Double]?
    var d_weights: [Double]?
    
    var input: [Double]?
    var outputs: [Double]?
    var activations: [Double]?
    
    var actFn: Activation
    
    init(inputSize: UInt, size:UInt, actFn:Activation) {
        self.inputSize = inputSize
        self.size = size;
        self.actFn = actFn;
        
        self.biases = makeRandomArray(n: size)
        self.weights = makeRandomArray(n: inputSize*size)
        
        self.d_biases = [Double](repeating : 0.0, count : Int(self.biases!.count))
        self.d_weights = [Double](repeating : 0.0, count : Int(self.weights!.count))
    }
    
    
    func compute(input:[Double], inputSize:UInt) -> ([Double],UInt) {
        var actResult = [Double](repeating : 0.0, count : Int(self.size))
        var multResult = [Double](repeating : 0.0, count : Int(self.size))
        
        vDSP_mmulD(self.weights!, 1, input, 1, &multResult, 1, self.size, 1, inputSize)
        vDSP_vaddD(multResult, 1, self.biases!, 1, &actResult, 1, self.size)
        
        self.input = input
        self.outputs = actResult
        self.activations = self.actFn.Compute(output: self.outputs!)
        return (self.activations!, self.size)
    }
    
    // For final layer only. Should add validation.
    func initError(cost: [Double]) -> [Double]
    {
        let sigmoidPrime = self.actFn.Delta(output: self.outputs!)
        var delta = [Double](repeating : 0.0, count : cost.count)
        vDSP_vmulD(cost, 1, sigmoidPrime, 1, &delta, 1, UInt(cost.count))
        
        var inputTranpose = [Double](repeating : 0.0, count : self.input!.count)
        vDSP_mtransD(self.input!, 1, &inputTranpose, 1, UInt(self.input!.count), 1)
        
        var inputMult = [Double](repeating : 0.0, count : Int(self.weights!.count))
        vDSP_mmulD(delta, 1, inputTranpose, 1, &inputMult, 1, self.size, UInt(inputTranpose.count), 1)
        
        var d_weights = [Double](repeating : 0.0, count : Int(self.weights!.count))
        vDSP_vaddD(delta, 1, inputMult, 1, &d_weights, 1, UInt(self.weights!.count))
        self.d_weights = d_weights
        
        return delta
    }
    
    func updateError(delta: [Double], nextLayer: Layer) -> [Double]
    {
        var transpose = [Double](repeating : 0.0, count : nextLayer.weights!.count)
        vDSP_mtransD(nextLayer.weights!, 1, &transpose, 1, nextLayer.size, self.size)
        
        var mult = [Double](repeating : 0.0, count : Int(self.size))
        vDSP_mmulD(transpose, 1, delta, 1, &mult, 1, self.size, 1, nextLayer.size)
        
        let sigmoidPrime = self.actFn.Delta(output: self.outputs!)
        var delta = [Double](repeating : 0.0, count : Int(self.size))
        vDSP_vmulD(mult, 1, sigmoidPrime, 1, &delta, 1, self.size)
        
        // Update Biases
        var d_biases = [Double](repeating : 0.0, count : Int(self.size))
        vDSP_vaddD(self.d_biases!, 1, delta, 1, &d_biases, 1, self.size)
        self.d_biases = d_biases
        
        // Update Weights
        var inputTranpose = [Double](repeating : 0.0, count : self.input!.count)
        vDSP_mtransD(self.input!, 1, &inputTranpose, 1, UInt(self.input!.count), 1)
        
        var inputMult = [Double](repeating : 0.0, count : Int(self.weights!.count))
        vDSP_mmulD(delta, 1, inputTranpose, 1, &inputMult, 1, self.size, UInt(inputTranpose.count), 1)
        
        var d_weights = [Double](repeating : 0.0, count : Int(self.d_weights!.count))
        vDSP_vaddD(self.d_weights!, 1, inputMult, 1, &d_weights, 1, UInt(self.d_weights!.count))
        self.d_weights = d_weights
        
        return delta
    }
    
    func updateWeights(miniBatchSize: Int, eta: Double)
    {
        var lambda = eta / Double(miniBatchSize)
        var deltaWeights = [Double](repeating : 0.0, count : self.weights!.count)
        var updatedWeights = [Double](repeating : 0.0, count : self.weights!.count)
        
        var deltaBiases = [Double](repeating : 0.0, count : self.biases!.count)
        var updatedBiases = [Double](repeating : 0.0, count : self.biases!.count)
        
        vDSP_vsmulD(self.d_weights!, 1, &lambda, &deltaWeights, 1, vDSP_Length(self.weights!.count))
        vDSP_vsubD(self.weights!, 1, deltaWeights, 1, &updatedWeights, 1, vDSP_Length(self.weights!.count))
        
        vDSP_vsmulD(self.d_weights!, 1, &lambda, &deltaBiases, 1, vDSP_Length(self.biases!.count))
        vDSP_vsubD(self.weights!, 1, deltaWeights, 1, &updatedBiases, 1, vDSP_Length(self.biases!.count))
        
        self.weights = updatedWeights
        self.biases = updatedBiases
        self.d_weights = [Double](repeating : 0.0, count : self.weights!.count)
        self.d_biases = [Double](repeating : 0.0, count : self.biases!.count)
    }
    
    func makeRandomArray(n: UInt) -> [Double] {
        var result:[Double] = []
        for _ in 0..<Int(n) {
            result.append(Double(arc4random_uniform(30)) / 30.0)
        }
        return result
    }
    
    
}
