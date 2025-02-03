import Accelerate
import ARKit
import CoreImage
import Flutter
import onnxruntime_objc
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
    var ortSession: ORTSession?
    var methodChannel: FlutterMethodChannel?

    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        let modelPath = Bundle.main.path(forResource: "OULU_Protocol_2_model_0_0", ofType: "onnx")!

        do {
            let ortEny = try ORTEnv(loggingLevel: .warning)
            ortSession = try ORTSession(env: ortEny, modelPath: modelPath, sessionOptions: nil)
        } catch {
            print(error)
        }

        let controller = window?.rootViewController as! FlutterViewController
        methodChannel = FlutterMethodChannel(name: "onnx", binaryMessenger: controller.binaryMessenger)

        methodChannel?.setMethodCallHandler { call, result in
            if call.method == "processImage" {
                guard let args = call.arguments as? [String: Any],
                      let imageData = args["data"] as? FlutterStandardTypedData,
                      let width = args["width"] as? Int,
                      let height = args["height"] as? Int,
                      let bytesPerRow = args["bytesPerRow"] as? Int
                else {
                    return
                }

                if let sampleBuffer = self.createCMSampleBufferFromBGRA(data: imageData.data,
                                                                        width: width,
                                                                        height: height,
                                                                        bytesPerRow: bytesPerRow),

                    let testData = CMSampleBufferGetImageBuffer(sampleBuffer)?.normalized(224, 224)?.withUnsafeBufferPointer({ Data(buffer: $0) })
                {
                    self.runOnnxModel(testData)
                } else {
                    result(FlutterError(code: "PROCESSING_ERROR", message: "Failed to create CMSampleBuffer", details: nil))
                }

            } else {
                result(FlutterMethodNotImplemented)
            }
        }

        GeneratedPluginRegistrant.register(with: self)
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }

    func runOnnxModel(_ imageData: Data) {
        guard let ortSession = ortSession
        else { return }

        do {
            let inputTensor = try ORTValue(tensorData: NSMutableData(data: imageData),
                                           elementType: .float,
                                           shape: [1, 3, 224, 224])
            let outputs = try ortSession.run(withInputs: ["input": inputTensor],
                                             outputNames: ["output_pixel", "output_binary"],
                                             runOptions: nil)

            guard let pixel = outputs["output_pixel"],
                  let binary = outputs["output_binary"],
                  let pixelData = try? pixel.tensorData() as Data,
                  let binaryData = try? binary.tensorData() as Data
            else {
                return
            }

            let pixels = pixelData.toFloatArray()
            let binaries = binaryData.toFloatArray()

            let meanPixel = pixels.reduce(0, +) / Float(pixels.count)
            let meanBinary = binaries.reduce(0, +) / Float(binaries.count)

            let variancePixel = pixels.map { pow($0 - meanPixel, 2) }.reduce(0, +) / Float(pixels.count)
            let stdDevPixel = sqrt(variancePixel)

            let varianceBinary = binaries.map { pow($0 - meanBinary, 2) }.reduce(0, +) / Float(binaries.count)
            let stdDevBinary = sqrt(varianceBinary)

            let score = (meanPixel + meanBinary) / 2.0

            print("=================")
            print(score)
            print(meanPixel)
            print(meanBinary)
            print(stdDevPixel)
            print(stdDevBinary)
            print("=================")

        } catch {
            print(error)
        }
    }

    func createCMSampleBufferFromBGRA(data: Data, width: Int, height: Int, bytesPerRow: Int) -> CMSampleBuffer? {
        var pixelBuffer: CVPixelBuffer?

        let status = data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> OSStatus in
            guard let baseAddress = ptr.baseAddress else { return -1 }
            return CVPixelBufferCreateWithBytes(
                nil,
                width,
                height,
                kCVPixelFormatType_32BGRA, // Use BGRA
                UnsafeMutableRawPointer(mutating: baseAddress),
                bytesPerRow,
                nil, // You can supply a release callback if needed
                nil,
                nil,
                &pixelBuffer
            )
        }

        if status != kCVReturnSuccess {
            print("Error creating CVPixelBuffer: \(status)")
            return nil
        }

        guard let pixelBuffer = pixelBuffer else { return nil }

        // Create a CMVideoFormatDescription from the pixel buffer.
        var formatDescription: CMFormatDescription?
        let formatStatus = CMVideoFormatDescriptionCreateForImageBuffer(allocator: nil, imageBuffer: pixelBuffer, formatDescriptionOut: &formatDescription)

        if formatStatus != noErr {
            print("Error creating format description: \(formatStatus)")
            return nil
        }

        // Set up sample timing (adjust as necessary)
        var timingInfo = CMSampleTimingInfo(
            duration: CMTime.invalid,
            presentationTimeStamp: CMTime.zero,
            decodeTimeStamp: CMTime.invalid
        )

        // Create the CMSampleBuffer
        var sampleBuffer: CMSampleBuffer?
        let sampleBufferStatus = CMSampleBufferCreateForImageBuffer(allocator: nil, imageBuffer: pixelBuffer, dataReady: true, makeDataReadyCallback: nil, refcon: nil, formatDescription: formatDescription!, sampleTiming: &timingInfo, sampleBufferOut: &sampleBuffer)
        if sampleBufferStatus != noErr {
            print("Error creating CMSampleBuffer: \(sampleBufferStatus)")
            return nil
        }

        return sampleBuffer
    }
}

extension CVPixelBuffer {
    func normalized(_ width: Int, _ height: Int) -> [Float]? {
        let srcWidth = CVPixelBufferGetWidth(self)
        let srcHeight = CVPixelBufferGetHeight(self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(self)
        let bytesPerPixel = 4

        // Center crop to square
        let cropSize = min(srcWidth, srcHeight)
        let cropX = (srcWidth - cropSize) / 2
        let cropY = (srcHeight - cropSize) / 2

        CVPixelBufferLockBaseAddress(self, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(self)?.advanced(by: cropY * bytesPerRow + cropX * bytesPerPixel) else { return nil }

        var inBuffer = vImage_Buffer(data: baseAddress,
                                     height: UInt(cropSize),
                                     width: UInt(cropSize),
                                     rowBytes: bytesPerRow)

        guard let dstData = malloc(width * height * bytesPerPixel) else { return nil }
        defer { free(dstData) }
        var outBuffer = vImage_Buffer(data: dstData,
                                      height: UInt(height),
                                      width: UInt(width),
                                      rowBytes: width * bytesPerPixel)
        let error = vImageScale_ARGB8888(&inBuffer, &outBuffer, nil, vImage_Flags(0))
        guard error == kvImageNoError else { return nil }

        // Normalize using Accelerate-friendly loop
        var normalizedBuffer = [Float](repeating: 0, count: width * height * 3)
        for i in 0 ..< width * height {
            // Note: Adjust offsets based on expected channel order (B, G, R)
            let pixelOffset = i * 4
            let r = (Float(dstData.load(fromByteOffset: pixelOffset + 2, as: UInt8.self)) / 255.0 - 0.485) / 0.229
            let g = (Float(dstData.load(fromByteOffset: pixelOffset + 1, as: UInt8.self)) / 255.0 - 0.456) / 0.224
            let b = (Float(dstData.load(fromByteOffset: pixelOffset, as: UInt8.self)) / 255.0 - 0.406) / 0.225
            normalizedBuffer[i] = r
            normalizedBuffer[width * height + i] = g
            normalizedBuffer[width * height * 2 + i] = b
        }
        return normalizedBuffer
    }
}

extension Data {
    func toFloatArray() -> [Float] {
        return withUnsafeBytes { pointer in
            Array(pointer.bindMemory(to: Float.self))
        }
    }
}
