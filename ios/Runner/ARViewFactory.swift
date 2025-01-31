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
    private var ortSession: ORTSession?

    private var sceneView: ARSCNView
    private var faceLivenessChannel: FlutterMethodChannel
    private var onnxChannel: FlutterMethodChannel

    private let ciContext = CIContext()

    // Flag to track ONNX processing state
    private var lastProcessTime: TimeInterval = 0
    private let processInterval: TimeInterval = 0.25
    private var isProcessing: Bool = false

    init(frame: CGRect, messenger: FlutterBinaryMessenger) {
        sceneView = ARSCNView(frame: frame)
        faceLivenessChannel = FlutterMethodChannel(name: "face_liveness", binaryMessenger: messenger)
        onnxChannel = FlutterMethodChannel(name: "onnx", binaryMessenger: messenger)
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
            let ortEnv = try ORTEnv(loggingLevel: .warning)
            let sessionOptions = try ORTSessionOptions()
            try sessionOptions.setIntraOpNumThreads(1) // Limit to 1 CPU thread
            ortSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: sessionOptions)
        } catch {
            print("ONNX model setup error: \(error)")
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

            DispatchQueue.main.async { [weak self] in
                self?.faceLivenessChannel.invokeMethod("lightData", arguments: data)
            }
        }

        let currentTime = CACurrentMediaTime()
        guard currentTime - lastProcessTime > processInterval else { return }
        lastProcessTime = currentTime

        guard !isProcessing else { return }
        isProcessing = true

        let pixelBuffer = frame.capturedImage

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            if let imageData = self.convertPixelBufferToData(pixelBuffer) {
                self.runOnnxModel(imageData)
            } else {
                DispatchQueue.main.async { self.isProcessing = false }
            }
        }
    }

    func convertPixelBufferToData(_ pixelBuffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        return UIImage(cgImage: cgImage).jpegData(compressionQuality: 1)
    }

    func runOnnxModel(_ imageData: Data) {
        do {
            let inputTensor = try ORTValue(tensorData: NSMutableData(data: imageData),
                                           elementType: .float,
                                           shape: [1, 3, 224, 224])

            let outputs = try ortSession?.run(withInputs: ["input": inputTensor],
                                              outputNames: ["output_pixel"],
                                              runOptions: nil)

            DispatchQueue.main.async { [weak self] in
                print(outputs)
                self?.onnxChannel.invokeMethod("processImage", arguments: outputs?.description ?? "")
                self?.isProcessing = false
            }
        } catch {
            print("ONNX model error: \(error)")
            DispatchQueue.main.async { self.isProcessing = false }
        }
    }
}

extension ARView: ARSCNViewDelegate {
    func renderer(_: SCNSceneRenderer, didUpdate _: SCNNode, for anchor: ARAnchor) {
        guard let faceAnchor = anchor as? ARFaceAnchor else { return }

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

        DispatchQueue.main.async { [weak self] in
            self?.faceLivenessChannel.invokeMethod("faceTrackingData", arguments: data)
        }
    }
}
