import 'package:arkit_face_demo/app_text_style.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const FaceTrackingPage(),
    );
  }
}

class FaceTrackingPage extends StatefulWidget {
  const FaceTrackingPage({super.key});

  @override
  State<FaceTrackingPage> createState() => _FaceTrackingPageState();
}

class _FaceTrackingPageState extends State<FaceTrackingPage> {
  static const MethodChannel _faceLivenessChannel =
      MethodChannel('face_liveness');
  static const MethodChannel _onnxChannel = MethodChannel('onnx');

  double _leftEyeBlink = 0;
  double _rightEyeBlink = 0;
  double _eyeLookInLeft = 0;
  double _eyeLookInRight = 0;
  double _stdDevDepth = 0;

  double _ambientIntensity = 0.0;
  double _ambientColorTemperature = 0.0;

  String? _processImageResult;

  Uint8List? _currentFaceImage;

  @override
  void initState() {
    super.initState();
    _faceLivenessChannel.setMethodCallHandler(_handleFaceLivenessChannel);
    _onnxChannel.setMethodCallHandler(_handleOnnxChannel);
  }

  Future<void> _handleFaceLivenessChannel(MethodCall call) async {
    if (call.method == 'faceTrackingData') {
      final arguments = call.arguments as Map<dynamic, dynamic>;

      final leftEyeBlink = arguments['leftEyeBlink'] as double;
      final rightEyeBlink = arguments['rightEyeBlink'] as double;
      final eyeLookInLeft = arguments['eyeLookInLeft'] as double;
      final eyeLookInRight = arguments['eyeLookInRight'] as double;
      final stdDevDepth = arguments['stdDevDepth'] as double;

      setState(() {
        _leftEyeBlink = leftEyeBlink;
        _rightEyeBlink = rightEyeBlink;
        _eyeLookInLeft = eyeLookInLeft;
        _eyeLookInRight = eyeLookInRight;
        _stdDevDepth = stdDevDepth;
      });
    } else if (call.method == 'lightData') {
      final arguments = call.arguments as Map<dynamic, dynamic>;

      final ambientIntensity = arguments['ambientIntensity'] as double;
      final ambientColorTemperature =
          arguments['ambientColorTemperature'] as double;

      setState(() {
        _ambientIntensity = ambientIntensity;
        _ambientColorTemperature = ambientColorTemperature;
      });
    }
  }

  Future<void> _handleOnnxChannel(MethodCall call) async {
    if (call.method == 'processImage') {
      setState(() {
        _processImageResult = call.arguments;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Stack(
          children: [
            const ARCameraView(),
            Align(
              alignment: Alignment.topCenter,
              child: Padding(
                padding: const EdgeInsets.only(top: 50.0),
                child: Column(
                  spacing: 4,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      'Left Eye Blink: ${_leftEyeBlink.toStringAsFixed(6)}',
                      style: _leftEyeBlink > 0.8
                          ? AppTextStyle.w600(16).colorLightGreen
                          : AppTextStyle.w400(16).colorWhite,
                    ),
                    Text(
                      'Right Eye Blink: ${_rightEyeBlink.toStringAsFixed(6)}',
                      style: _rightEyeBlink > 0.8
                          ? AppTextStyle.w600(16).colorLightGreen
                          : AppTextStyle.w400(16).colorWhite,
                    ),
                    Text(
                      'Eye Look In Left: ${_eyeLookInLeft.toStringAsFixed(6)}',
                      style: _eyeLookInLeft > 0.8
                          ? AppTextStyle.w600(16).colorLightGreen
                          : AppTextStyle.w400(16).colorWhite,
                    ),
                    Text(
                      'Eye Look In Right: ${_eyeLookInRight.toStringAsFixed(6)}',
                      style: _eyeLookInRight > 0.8
                          ? AppTextStyle.w600(16).colorLightGreen
                          : AppTextStyle.w400(16).colorWhite,
                    ),
                    Text(
                      'Std.Dev Depth: ${_stdDevDepth.toStringAsFixed(6)}',
                      style: AppTextStyle.w400(16).colorWhite,
                    ),
                    Text(
                      'Ambient Intensity: ${_ambientIntensity.toStringAsFixed(6)}',
                      style: AppTextStyle.w400(16).colorWhite,
                    ),
                    Text(
                      'Ambient Color Temperature: ${_ambientColorTemperature.toStringAsFixed(6)}',
                      style: AppTextStyle.w400(16).colorWhite,
                    ),
                    Text(
                      'Process Image Result: $_processImageResult',
                      style: AppTextStyle.w400(16).colorWhite,
                    ),
                  ],
                ),
              ),
            ),
            if (_currentFaceImage != null)
              Positioned(
                bottom: 0,
                right: 0,
                child: Image.memory(
                  _currentFaceImage!,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class ARCameraView extends StatelessWidget {
  const ARCameraView({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.black,
      child: const UiKitView(
        viewType: 'ar_view',
        creationParams: null,
        creationParamsCodec: StandardMessageCodec(),
      ),
    );
  }
}
