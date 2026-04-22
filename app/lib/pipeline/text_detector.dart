// text_detector.dart
// DBNet-MobileNetV3 text detector. Returns list of rotated polygons.

import 'dart:typed_data';
import 'package:image/image.dart' as img;

import '../core/tflite_base.dart';

class TextPolygon {
  final List<Point> points;   // 4 corners of rotated rect
  final double score;
  const TextPolygon(this.points, this.score);
}

class Point {
  final double x, y;
  const Point(this.x, this.y);
}

class TextDetectorService extends TfliteBase {
  static const int _inputSize = 640;

  @override
  String get assetName => 'text_det_int8.tflite';

  @override
  List<int> get inputShape => const [1, _inputSize, _inputSize, 3];

  @override
  Object buildWarmupInput() => Uint8List(_inputSize * _inputSize * 3);

  Future<List<TextPolygon>> detect(img.Image doc) async {
    // TODO(week 2):
    //   1. Preprocess to 640×640 INT8 tensor
    //   2. Run interpreter → probability map [1, 640, 640, 1]
    //   3. Threshold at 0.3
    //   4. Contour-find → min-area-rect per connected region
    //   5. Unclip polygon (DB paper Eq. after binarization)
    //   6. Rescale to doc coords
    throw UnimplementedError('Implement in week 2.');
  }
}
