import ARKit
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
        let controller = window?.rootViewController as! FlutterViewController
        let registrar = controller.registrar(forPlugin: "ARViewFactory")
        registrar!.register(ARViewFactory(messenger: controller.binaryMessenger), withId: "ar_view")
        //        methodChannel = FlutterMethodChannel(name: "onnx", binaryMessenger: controller.binaryMessenger)

//        let modelPath = Bundle.main.path(forResource: "OULU_Protocol_2_model_0_0", ofType: "onnx")!
//
//        do {
//            let ortEny = try ORTEnv(loggingLevel: .warning)
//            ortSession = try ORTSession(env: ortEny, modelPath: modelPath, sessionOptions: nil)
//        } catch {
//            print(error);
//        }
//
//
//        methodChannel?.setMethodCallHandler { call, result in
//            if call.method == "processImage" {
//                guard let args = call.arguments as? [String: Any],
//                      let imageData = args["image"] as? FlutterStandardTypedData
//                else {
//                    result(FlutterError(code: "INVALID_ARGUMENTS", message: "Invalid image data", details: nil))
//                    return
//                }
//
//                self.runOnnxModel(imageData.data)
//
//            } else {
//                result(FlutterMethodNotImplemented)
//            }
//        }

        GeneratedPluginRegistrant.register(with: self)
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }

    func runOnnxModel(_ imageData: Data) {
        do {
            let inputTensor = try ORTValue(tensorData: NSMutableData(data: imageData),
                                           elementType: .float,
                                           shape: [1, 3, 224, 224])
            let outputs = try ortSession?.run(withInputs: ["input": inputTensor],
                                              outputNames: ["output_pixel"],
                                              runOptions: nil)

            print(outputs)
            methodChannel?.invokeMethod("processImage", arguments: outputs?.description ?? "")
        } catch {
            print(error)
        }
    }
}
