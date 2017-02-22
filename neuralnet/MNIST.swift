//
//  MNISTLoader.swift
//  neuralnet
//
//  Created by Tony Francis on 2/21/17.
//  Copyright © 2017 Tony Francis. All rights reserved.
//  Inspired by: https://github.com/timestocome/ReadMNIST/blob/master/Input%20MNIST/ReadMNIST.swift
//

import Foundation

class MNIST {

    func loadData(imagePath: String, labelPath: String) -> DataList {
        // Targets
        let labelRawData = NSData(contentsOfFile: labelPath)
        
        let labelLength = labelRawData?.length       // number of bytes in data
        var labelData:[UInt8] = Array(repeating: 0, count: labelLength!)
        labelRawData?.getBytes(&labelData, length: labelLength!)
        labelData.removeSubrange(0..<8)
        
        var targets:[[Double]] = labelData.map { (n:UInt8) in
            var t = [Double](repeating: 0.0, count: 10)
            t[Int(n)] = 1.0
            return t
        }
        
        // Inputs
        let imageRawData = NSData(contentsOfFile: imagePath)
        
        let imageLength = imageRawData?.length       // number of bytes in data
        var imageData:[UInt8] = Array(repeating: 0, count: imageLength!)
        imageRawData?.getBytes(&imageData, length: imageLength!)
        imageData.removeSubrange(0..<16)
        
        var doubleData = Array(repeating: 0.0, count: imageData.count)
        for i in 0..<imageData.count {
            doubleData[i] = Double(imageData[i])
        }
        
        var inputs:[[Double]] = Array()
        for i in stride(from: 0, to: imageData.count, by: 784) {
            inputs.append(Array(doubleData[i..<min(i+784,imageData.count)]))
        }
        
        var data:Data = Array()
        for i in 0..<inputs.count {
            data.append((inputs[i], targets[i]))
        }

        return DataList(data: data, num: UInt(targets.count))
    }
    
    
    
}
