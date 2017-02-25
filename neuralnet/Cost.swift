//
//  Cost.swift
//  neuralnet
//
//  Created by Tony Francis on 2/21/17.
//  Copyright © 2017 Tony Francis. All rights reserved.
//

import Foundation
import Accelerate


protocol Cost {
    func Delta(output: [Double], target: [Double]) -> [Double]
}

class Quad: Cost {
    internal func Delta(output: [Double], target: [Double]) -> [Double] {
        var result = [Double](repeating : 0.0, count : output.count)
        vDSP_vsubD(output, 1, target, 1, &result, 1, UInt(output.count))
        return result
    }
}
