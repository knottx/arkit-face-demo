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

        GeneratedPluginRegistrant.register(with: self)
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }
}
