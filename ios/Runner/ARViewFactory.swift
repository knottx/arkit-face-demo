//
//  ARViewFactory.swift
//  Runner
//
//  Created by Visarut Tippun on 30/1/2568 BE.
//

import ARKit
import CoreML
import Flutter
import onnxruntime_objc
import UIKit

class ARViewFactory: NSObject, FlutterPlatformViewFactory {
    private var messenger: FlutterBinaryMessenger

    init(messenger: FlutterBinaryMessenger) {
        self.messenger = messenger
    }

    func create(
        withFrame frame: CGRect,
        viewIdentifier _: Int64,
        arguments _: Any?
    ) -> FlutterPlatformView {
        return ARView(frame: frame, messenger: messenger)
    }
}

class ARView: NSObject, FlutterPlatformView {
    var ortSession: ORTSession?

    var sceneView: ARSCNView
    var methodChannel: FlutterMethodChannel

    init(frame: CGRect, messenger: FlutterBinaryMessenger) {
        sceneView = ARSCNView(frame: frame)
        methodChannel = FlutterMethodChannel(name: "face_liveness", binaryMessenger: messenger)
        super.init()
        setupOnnxModel()

        let configuration = ARFaceTrackingConfiguration()
        configuration.isLightEstimationEnabled = true
        sceneView.delegate = self
        sceneView.session.run(configuration)
        sceneView.session.delegate = self
    }

    func view() -> UIView {
        return sceneView
    }

    deinit {
        self.sceneView.session.pause()
    }

    func setupOnnxModel() {
        guard let modelPath = Bundle.main.path(forResource: "OULU_Protocol_2_model_0_0", ofType: "onnx") else { return }

        do {
            let ortEny = try ORTEnv(loggingLevel: .warning)
            ortSession = try ORTSession(env: ortEny, modelPath: modelPath, sessionOptions: nil)
        } catch {
            print(error)
        }
    }
}

extension ARView: ARSessionDelegate {
    func session(_: ARSession, didUpdate frame: ARFrame) {
        if let lightEstimate = frame.lightEstimate {
            let ambientIntensity = lightEstimate.ambientIntensity
            let ambientColorTemperature = lightEstimate.ambientColorTemperature

            let data: [String: Any] = [
                "ambientIntensity": ambientIntensity,
                "ambientColorTemperature": ambientColorTemperature,
            ]
            methodChannel.invokeMethod("lightData", arguments: data)
        }
    }
}

extension ARView: ARSCNViewDelegate {
    func renderer(_: SCNSceneRenderer, didUpdate _: SCNNode, for anchor: ARAnchor) {
        guard let faceAnchor = anchor as? ARFaceAnchor else { return }

//        self.runModel(with: faceAnchor)

        let blendShapes = faceAnchor.blendShapes
        let leftEyeBlink = blendShapes[.eyeBlinkLeft]?.floatValue ?? 0.0
        let rightEyeBlink = blendShapes[.eyeBlinkRight]?.floatValue ?? 0.0
        let eyeLookInLeft = blendShapes[.eyeLookInLeft]?.floatValue ?? 0.0
        let eyeLookInRight = blendShapes[.eyeLookInRight]?.floatValue ?? 0.0

        let faceGeometry = faceAnchor.geometry
        let vertices = faceGeometry.vertices
        let depths = vertices.map { $0.z }
        let avgDepth = depths.reduce(0, +) / Float(depths.count)
        let variance = depths.map { pow($0 - avgDepth, 2) }.reduce(0, +) / Float(depths.count)
        let stdDev = sqrt(variance)

        sendDataToFlutter(leftEyeBlink: leftEyeBlink,
                          rightEyeBlink: rightEyeBlink,
                          eyeLookInLeft: eyeLookInLeft,
                          eyeLookInRight: eyeLookInRight,
                          stdDevDepth: stdDev)
    }

    private func sendDataToFlutter(leftEyeBlink: Float,
                                   rightEyeBlink: Float,
                                   eyeLookInLeft: Float,
                                   eyeLookInRight: Float,
                                   stdDevDepth: Float)
    {
        let data: [String: Any] = [
            "leftEyeBlink": leftEyeBlink,
            "rightEyeBlink": rightEyeBlink,
            "eyeLookInLeft": eyeLookInLeft,
            "eyeLookInRight": eyeLookInRight,
            "stdDevDepth": stdDevDepth,
        ]

        methodChannel.invokeMethod("faceTrackingData", arguments: data)
    }

    func convertBlendShapesToMLMultiArray(arFaceAnchor: ARFaceAnchor) -> MLMultiArray? {
        // Extract blend shapes from ARFaceAnchor
        let blendShapes = arFaceAnchor.blendShapes

        // Flatten blend shape values into an array of Float values
        let blendShapeValues = blendShapes.values.map { $0.floatValue }

        // Create MLMultiArray to hold blend shape values
        do {
            // Create MLMultiArray with shape [1, N] where N is the number of blend shapes
            let blendShapeArray = try MLMultiArray(shape: [1, NSNumber(value: blendShapeValues.count)], dataType: .float32)

            // Populate MLMultiArray with values
            for (index, value) in blendShapeValues.enumerated() {
                blendShapeArray[index] = NSNumber(value: value)
            }

            return blendShapeArray
        } catch {
            print("Failed to create MLMultiArray: \(error)")
            return nil
        }
    }

    func runModel(with arFaceAnchor: ARFaceAnchor) {
        guard let ortSession = ortSession else {
            print("ORTSession is not initialized.")
            return
        }

        // Convert blend shapes to MLMultiArray
        guard let inputData = convertBlendShapesToMLMultiArray(arFaceAnchor: arFaceAnchor) else {
            print("Failed to convert blend shapes to MLMultiArray.")
            return
        }

        do {
            // Convert MLMultiArray to Data
//            let inputTensorData = try convertMLMultiArrayToData(inputData)
//
//            // Create an ORTValue with the converted Data
//                   let inputName = "input"
//                   let inputTensor = try ORTValue(
//                       tensorData: NSMutableData(data: inputTensorData),  // Data format
//                       elementType: ORTTensorElementDataType.float,  // Type: Float for MLMultiArray
//                       shape: [1, inputTensor.count]  // Adjust shape as per your model input
//                   )
//
//           let outputs = try ortSession.run(
//               withInputs: [inputName: inputTensor],
//               outputNames: ["output_pixel"],
//               runOptions: nil
//           )
            ////
//           debugPrint("XXXXXXXXXXXXXX")
//           debugPrint(outputs)
//           debugPrint("XXXXXXXXXXXXXX")
//
//            let result = calculateLivenessScore(outputs: outputs)
        } catch {
            print("Failed to run model: \(error)")
            print(error)
        }
    }

    // Function to convert MLMultiArray to Data
    func convertMLMultiArrayToData(_ multiArray: MLMultiArray) throws -> Data {
        let count = multiArray.count
        let pointer = UnsafeMutableBufferPointer(start: multiArray.dataPointer.assumingMemoryBound(to: Float.self), count: count)
        let data = Data(bytes: pointer.baseAddress!, count: count * MemoryLayout<Float>.size)

        return data
    }

//    private func calculateLivenessScore(outputs: [String: ORTValue]) -> Float? {
//        guard let outputPixelTensor = outputs["output_pixel"],
//              let outputBinaryTensor = outputs["output_binary"] else { return nil }
//
//        let outputPixelArray = outputPixelTensor.toFloatArray()
//        let outputBinaryArray = outputBinaryTensor.toFloatArray()
//
//        let pixelMean = outputPixelArray.reduce(0, +) / Float(outputPixelArray.count)
//        let binaryMean = outputBinaryArray.reduce(0, +) / Float(outputBinaryArray.count)
//        return (pixelMean + binaryMean) / 2.0
//    }
}

extension CVPixelBuffer {
    func toFloatArray() -> [Float] {
        CVPixelBufferLockBaseAddress(self, .readOnly)
        let width = CVPixelBufferGetWidth(self)
        let height = CVPixelBufferGetHeight(self)
        let baseAddress = CVPixelBufferGetBaseAddress(self)

        let buffer = baseAddress?.assumingMemoryBound(to: UInt8.self)
        var floatArray: [Float] = []

        for y in 0 ..< height {
            for x in 0 ..< width {
                let offset = (y * width + x) * 4
                let r = Float(buffer?[offset + 1] ?? 0) / 255.0
                let g = Float(buffer?[offset + 2] ?? 0) / 255.0
                let b = Float(buffer?[offset + 3] ?? 0) / 255.0

                floatArray.append(r)
                floatArray.append(g)
                floatArray.append(b)
            }
        }

        CVPixelBufferUnlockBaseAddress(self, .readOnly)
        return floatArray
    }

//    private func preprocessImage(_ image: UIImage) -> Data? {
//            guard let resizedImage = image.resized(to: CGSize(width: 224, height: 224)),
//                  let pixelBuffer = resizedImage.pixelBuffer() else {
//                return nil
//            }
//
//            let floatArray = pixelBuffer.toFloatArray()
//            print("Float Array:", floatArray)
//            return floatArray.withUnsafeBytes { Data($0) }  // ✅ ใช้วิธีนี้เพื่อความปลอดภัย
//        }
//    private func extractFace(from image: CIImage, faceBoundingBox: CGRect) {
//            let faceRect = VNImageRectForNormalizedRect(faceBoundingBox, Int(image.extent.width), Int(image.extent.height))
//            let croppedImage = image.cropped(to: faceRect)
//
//            let context = CIContext()
//            if let cgImage = context.createCGImage(croppedImage, from: croppedImage.extent) {
//                let uiImage = UIImage(cgImage: cgImage)
//                DispatchQueue.main.async {
//                    self.faceImageView.image = uiImage
//                }
//
//                if let inputData = preprocessImage(uiImage) {
//                    let result = runModel(with: inputData)
//                    print("Model Output:", result ?? [])
//                }
//            }
//        }
}
