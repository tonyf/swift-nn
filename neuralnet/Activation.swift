//
//  Activation.swift
//  neuralnet
//
//  Created by Tony Francis on 2/21/17.
//  Copyright © 2017 Tony Francis. All rights reserved.
//

import Foundation

protocol Activation {
    func Compute(output: [Double]) -> [Double]
    func Delta(output: [Double]) -> [Double]
}

class Sigmoid: Activation {
    internal func Compute(output: [Double]) -> [Double] {
        return output.map{ z in 1.0 / (1.0 + exp(-z)) }
    }
    
    internal func Delta(output: [Double]) -> [Double]
    {
        return output.map{ (z:Double) in
                let sig = 1.0 / (1.0 + exp(-z))
                return sig * (1-sig)
        }
    }
}

class ReLU: Activation {
    internal func Compute(output: [Double]) -> [Double] {
        return output.map{ z in max(0, z) }
    }
    
    internal func Delta(output: [Double]) -> [Double]
    {
        return output.map{ (z:Double) in
            return z > 0 ? z : 0
        }
    }
}
