import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

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
  static const MethodChannel _faceLivenessChannel = MethodChannel('face_liveness');
  static const MethodChannel _onnxChannel = MethodChannel('onnx');

  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(),
  );

  DateTime? _lastProcessAt;

  final _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  bool _isBusy = false;

  CameraController? _cameraController;

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
    _startLiveFeed();
  }

  void _startLiveFeed() async {
    final cameras = await availableCameras();

    final frontCamera = cameras.firstWhere(
      (e) => e.lensDirection == CameraLensDirection.front,
    );

    _cameraController = CameraController(
      frontCamera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid ? ImageFormatGroup.nv21 : ImageFormatGroup.bgra8888,
    );

    await _cameraController?.initialize();

    await _cameraController?.startImageStream(
      (image) => _processCameraImage(
        image: image,
        camera: frontCamera,
      ),
    );

    setState(() {});
  }

  void _processCameraImage({
    required CameraImage image,
    required CameraDescription camera,
  }) async {
    if (_isBusy) return;
    _isBusy = true;

    final lastProcessAt = _lastProcessAt;
    if (lastProcessAt != null) {
      final now = DateTime.now();
      if (lastProcessAt.difference(now).inMilliseconds.abs() < 100) return;
      _lastProcessAt = now;
    }

    final sensorOrientation = camera.sensorOrientation;

    InputImageRotation? rotation;
    if (Platform.isIOS) {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    } else if (Platform.isAndroid) {
      final deviceOrientation = _cameraController?.value.deviceOrientation;
      int? rotationCompensation = (deviceOrientation != null) ? _orientations[deviceOrientation] : null;
      if (rotationCompensation == null) return;
      rotationCompensation = (sensorOrientation + rotationCompensation) % 360;
      rotation = InputImageRotationValue.fromRawValue(rotationCompensation);
    }

    if (rotation == null) return;

    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    // validate format depending on platform
    // only supported formats:
    // * nv21 for Android
    // * bgra8888 for iOS
    if (format == null ||
        (Platform.isAndroid && format != InputImageFormat.nv21) ||
        (Platform.isIOS && format != InputImageFormat.bgra8888)) {
      return;
    }

    // since format is constraint to nv21 or bgra8888, both only have one plane
    if (image.planes.length != 1) return;
    final plane = image.planes.first;

    final WriteBuffer allBytes = WriteBuffer();
    for (var plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }

    final inputImage = InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(
          image.width.toDouble(),
          image.height.toDouble(),
        ),
        rotation: rotation, // used only in Android
        format: format, // used only in iOS
        bytesPerRow: plane.bytesPerRow, // used only in iOS
      ),
    );

    final List<Face> faces = await _faceDetector.processImage(inputImage);
    if (faces.length == 1) {
      // final originalImage = cameraImageToImage(image);

      final Uint8List data = image.planes[0].bytes;
      final int width = image.width;
      final int height = image.height;
      final int bytesPerRow = image.planes[0].bytesPerRow;

      final Map<String, dynamic> args = {
        'data': data,
        'width': width,
        'height': height,
        'bytesPerRow': bytesPerRow,
      };

      _onnxChannel.invokeMethod(
        'processImage',
        args,
      );
    }

    _isBusy = false;
  }

  Float32List convertCameraImageToOnnxTensor(CameraImage image) {
    final int width = image.width;
    final int height = image.height;

    // Create a Float32List to hold our tensor data (3 channels, channel-first order).
    final int tensorSize = 3 * width * height;
    final Float32List tensor = Float32List(tensorSize);

    // Retrieve the Y, U, and V planes.
    final Plane planeY = image.planes[0];
    final Plane planeU = image.planes[1];
    final Plane planeV = image.planes[2];

    final int yRowStride = planeY.bytesPerRow;
    final int uvRowStride = planeU.bytesPerRow;
    final int uvPixelStride = planeU.bytesPerPixel ?? 1; // Usually 2, but default to 1 if null

    // Loop over every pixel.
    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        // --- Get the Y value for the current pixel ---
        final int yIndex = row * yRowStride + col;
        final int Y = planeY.bytes[yIndex];

        // --- Get the U and V values (subsampled: one value per 2x2 block) ---
        final int uvRow = row ~/ 2;
        final int uvCol = col ~/ 2;
        final int uvIndex = uvRow * uvRowStride + uvCol * uvPixelStride;
        final int U = planeU.bytes[uvIndex];
        final int V = planeV.bytes[uvIndex];

        // Convert YUV to RGB.
        // Note: U and V are offset by 128.
        double yVal = Y.toDouble();
        double uVal = U.toDouble() - 128.0;
        double vVal = V.toDouble() - 128.0;

        double r = yVal + 1.402 * vVal;
        double g = yVal - 0.344136 * uVal - 0.714136 * vVal;
        double b = yVal + 1.772 * uVal;

        // Clamp RGB values to [0, 255].
        r = r.clamp(0.0, 255.0);
        g = g.clamp(0.0, 255.0);
        b = b.clamp(0.0, 255.0);

        // Normalize values to [0, 1].
        double rNorm = r / 255.0;
        double gNorm = g / 255.0;
        double bNorm = b / 255.0;

        // Place the normalized values into the tensor.
        // Assuming channel-first order: [1, 3, height, width]
        int indexR = 0 * (width * height) + row * width + col;
        int indexG = 1 * (width * height) + row * width + col;
        int indexB = 2 * (width * height) + row * width + col;

        tensor[indexR] = rNorm;
        tensor[indexG] = gNorm;
        tensor[indexB] = bNorm;
      }
    }

    return tensor;
  }

  img.Image cameraImageToImage(CameraImage cameraImage) {
    // The width and height of the image from the camera
    int width = cameraImage.width;
    int height = cameraImage.height;

    // Create a list for RGB data
    List<int> rgbBytes = List.filled(width * height * 3, 0);

    // Get the BGRA plane
    var bgraPlane = cameraImage.planes[0];

    // Process each pixel
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // Calculate pixel position in the plane
        int pixelIndex = y * bgraPlane.bytesPerRow + x * 4;
        int blue = bgraPlane.bytes[pixelIndex];
        int green = bgraPlane.bytes[pixelIndex + 1];
        int red = bgraPlane.bytes[pixelIndex + 2];
        // int alpha = bgraPlane.bytes[pixelIndex + 3];

        // Convert BGRA to RGB (ignore alpha if you don't need it)
        int r = red;
        int g = green;
        int b = blue;

        // Store RGB values in the list
        int index = (y * width + x) * 3;
        rgbBytes[index] = r;
        rgbBytes[index + 1] = g;
        rgbBytes[index + 2] = b;
      }
    }

    // Create an img.Image object from the RGB byte list
    img.Image image = img.Image.fromBytes(
      width: width,
      height: height,
      bytes: Uint8List.fromList(rgbBytes).buffer,
    );

    return image;
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
      final ambientColorTemperature = arguments['ambientColorTemperature'] as double;

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
            // const ARCameraView(),
            if (_cameraController != null && _cameraController!.value.isInitialized) CameraPreview(_cameraController!),
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
                      style: TextStyle(color: Colors.white),
                    ),
                    Text(
                      'Right Eye Blink: ${_rightEyeBlink.toStringAsFixed(6)}',
                      style: TextStyle(color: Colors.white),
                    ),
                    Text(
                      'Eye Look In Left: ${_eyeLookInLeft.toStringAsFixed(6)}',
                      style: TextStyle(color: Colors.white),
                    ),
                    Text(
                      'Eye Look In Right: ${_eyeLookInRight.toStringAsFixed(6)}',
                      style: TextStyle(color: Colors.white),
                    ),
                    Text(
                      'Std.Dev Depth: ${_stdDevDepth.toStringAsFixed(6)}',
                      style: TextStyle(color: Colors.white),
                    ),
                    Text(
                      'Ambient Intensity: ${_ambientIntensity.toStringAsFixed(6)}',
                      style: TextStyle(color: Colors.white),
                    ),
                    Text(
                      'Ambient Color Temperature: ${_ambientColorTemperature.toStringAsFixed(6)}',
                      style: TextStyle(color: Colors.white),
                    ),
                    Text(
                      'Process Image Result: $_processImageResult',
                      style: TextStyle(color: Colors.white),
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
