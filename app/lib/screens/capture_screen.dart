import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

import '../pipeline/corner_detector.dart';
import '../pipeline/perspective_warp.dart';
import '../pipeline/doc_classifier.dart';

class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  bool _isInitialized = false;
  bool _hasCamera = false;
  bool _isProcessing = false;
  CameraController? _controller;
  late CornerDetectorService _cornerDetector;
  late DocClassifierService _docClassifier;
  String? _prediction;
  String? _pipelineStatus;
  final Map<String, int> _timings = {};

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    if (!mounted) return;
    setState(() {
      _pipelineStatus = 'Initializing services...';
    });

    _cornerDetector = CornerDetectorService();
    _docClassifier = DocClassifierService();

    try {
      await _cornerDetector.initialize();
      await _docClassifier.initialize();
    } catch (e) {
      debugPrint('Failed to initialize pipeline services: $e');
    }

    if (!mounted) return;

    try {
      final cameras = await availableCameras();
      if (cameras.isNotEmpty) {
        CameraDescription? backCam;
        for (final cam in cameras) {
          if (cam.lensDirection == CameraLensDirection.back) {
            backCam = cam;
            break;
          }
        }
        final selectedCam = backCam ?? cameras.first;
        _controller = CameraController(
          selectedCam,
          ResolutionPreset.high,
          enableAudio: false,
        );
        await _controller!.initialize();
        _hasCamera = true;
      } else {
        debugPrint('No cameras found.');
      }
    } catch (e) {
      debugPrint('Camera initialization failed: $e');
    }

    if (!mounted) return;

    setState(() {
      _isInitialized = true;
      _pipelineStatus = _hasCamera ? 'Ready' : 'Ready (No Camera Fallback)';
    });
  }

  Future<void> _captureAndProcess() async {
    if (!_isInitialized || _isProcessing) return;

    setState(() {
      _isProcessing = true;
      _prediction = null;
      _pipelineStatus = 'Processing...';
      _timings.clear();
    });

    final totalStopwatch = Stopwatch()..start();
    img.Image? originalImage;

    try {
      if (_hasCamera && _controller != null) {
        final captureStopwatch = Stopwatch()..start();
        final XFile xFile = await _controller!.takePicture();
        captureStopwatch.stop();
        _timings['Capture'] = captureStopwatch.elapsedMilliseconds;

        final decodeStopwatch = Stopwatch()..start();
        final bytes = await xFile.readAsBytes();
        originalImage = img.decodeImage(bytes);
        decodeStopwatch.stop();
        _timings['Decode'] = decodeStopwatch.elapsedMilliseconds;
      } else {
        // Procedural image fallback for simulation/testing
        final captureStopwatch = Stopwatch()..start();
        // Simulate physical capture time
        await Future.delayed(const Duration(milliseconds: 100));
        captureStopwatch.stop();
        _timings['Capture (Mock)'] = captureStopwatch.elapsedMilliseconds;

        final decodeStopwatch = Stopwatch()..start();
        // Generate a mock document image of 800x600 pixels
        originalImage = img.Image(width: 800, height: 600);
        // Draw a simulated dark background with a white card in the center
        img.fillRect(
          originalImage,
          x1: 0,
          y1: 0,
          x2: 800,
          y2: 600,
          color: img.ColorRgb8(30, 30, 30),
        );
        img.fillRect(
          originalImage,
          x1: 100,
          y1: 100,
          x2: 700,
          y2: 500,
          color: img.ColorRgb8(240, 240, 240),
        );
        decodeStopwatch.stop();
        _timings['Generate (Mock)'] = decodeStopwatch.elapsedMilliseconds;
      }

      if (originalImage == null) {
        throw Exception('Failed to acquire source image.');
      }

      // 1. Corner detection
      final cornerStopwatch = Stopwatch()..start();
      final corners = await _cornerDetector.detect(originalImage);
      cornerStopwatch.stop();
      _timings['Corner Detection'] = cornerStopwatch.elapsedMilliseconds;

      // 2. Perspective warp
      final warpStopwatch = Stopwatch()..start();
      final warpedImage = PerspectiveWarp.warp(originalImage, corners);
      warpStopwatch.stop();
      _timings['Perspective Warp'] = warpStopwatch.elapsedMilliseconds;

      // 3. Document classification
      final classifyStopwatch = Stopwatch()..start();
      final docType = await _docClassifier.classify(warpedImage);
      classifyStopwatch.stop();
      _timings['Classification'] = classifyStopwatch.elapsedMilliseconds;

      totalStopwatch.stop();
      _timings['Total End-to-End'] = totalStopwatch.elapsedMilliseconds;

      final resultStr = docType == DocType.aadhaar ? 'Aadhaar' : 'PAN';

      // Print all inference timings to console
      print('--- BharatScan Pipeline Timings ---');
      _timings.forEach((key, val) {
        print('$key: $val ms');
      });
      print('Prediction: $resultStr');
      print('----------------------------------');

      if (!mounted) return;

      setState(() {
        _prediction = resultStr;
        _pipelineStatus = 'Success';
      });
    } catch (e, stack) {
      debugPrint('Pipeline processing failed: $e\n$stack');
      if (!mounted) return;
      setState(() {
        _pipelineStatus = 'Failed: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _cornerDetector.dispose();
    _docClassifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('BharatScan'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Stack(
        children: [
          Positioned.fill(
            child: _isInitialized
                ? (_hasCamera && _controller != null
                    ? CameraPreview(_controller!)
                    : Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(
                              Icons.camera_alt,
                              size: 64,
                              color: Colors.grey,
                            ),
                            const SizedBox(height: 16),
                            Text(
                              _pipelineStatus ?? 'No camera available',
                              style: const TextStyle(
                                fontSize: 16,
                                color: Colors.grey,
                              ),
                              textAlign: TextAlign.center,
                            ),
                          ],
                        ),
                      ))
                : const Center(child: CircularProgressIndicator()),
          ),
          if (_prediction != null)
            Positioned(
              bottom: 100,
              left: 20,
              right: 20,
              child: Card(
                color: Colors.black87,
                elevation: 4,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                    vertical: 16,
                    horizontal: 24,
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Text(
                        'Prediction',
                        style: TextStyle(color: Colors.white70, fontSize: 14),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        _prediction!,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          if (_pipelineStatus != null &&
              _pipelineStatus != 'Ready' &&
              _pipelineStatus != 'Ready (No Camera Fallback)' &&
              _prediction == null)
            Positioned(
              top: 20,
              left: 20,
              right: 20,
              child: Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    vertical: 8,
                    horizontal: 16,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    _pipelineStatus!,
                    style: const TextStyle(color: Colors.white, fontSize: 14),
                  ),
                ),
              ),
            ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _isInitialized && !_isProcessing ? _captureAndProcess : null,
        tooltip: 'Capture',
        child: _isProcessing
            ? const SizedBox(
                width: 24,
                height: 24,
                child: CircularProgressIndicator(
                  strokeWidth: 2.5,
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                ),
              )
            : const Icon(Icons.camera),
      ),
    );
  }
}
