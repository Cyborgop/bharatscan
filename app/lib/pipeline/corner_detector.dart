// corner_detector.dart
// Runs the U-Net-lite corner heatmap model, returns 4 points.

import 'dart:typed_data';
import 'package:image/image.dart' as img;

import '../core/tflite_base.dart';

class Corner {
  final double x, y;
  const Corner(this.x, this.y);
  @override
  String toString() => '(${x.toStringAsFixed(1)}, ${y.toStringAsFixed(1)})';
}

class CornerDetectorService extends TfliteBase {
  static const int _inputSize = 256;
  static const int _heatSize = 64;

  @override
  String get assetName => 'corner_detector_int8.tflite';

  @override
  List<int> get inputShape => const [1, _inputSize, _inputSize, 3];

  // ImageNet mean/std
  static const double _rm = 0.485, _gm = 0.456, _bm = 0.406;
  static const double _rs = 0.229, _gs = 0.224, _bs = 0.225;

  // TODO: read from tensor.params at runtime
  static const double _qScale = 0.01865845;
  static const int _qZero = 114;

  @override
  Object buildWarmupInput() => Uint8List(_inputSize * _inputSize * 3);

  /// Returns [topLeft, topRight, bottomRight, bottomLeft] in image coords.
  Future<List<Corner>> detect(img.Image frame) async {
    final resized = img.copyResize(frame,
        width: _inputSize, height: _inputSize,
        interpolation: img.Interpolation.linear);

    final input = [_preprocess(resized)];
    final heatmaps = List.generate(
      1,
      (_) => List.generate(
        _heatSize,
        (_) => List.generate(_heatSize, (_) => List.filled(4, 0)),
      ),
    );
    print(input.runtimeType);
    print(input);
    await runMany(input, {0: heatmaps});

    // Argmax per channel
    final corners = <Corner>[];
    final scaleX = frame.width / _heatSize;
    final scaleY = frame.height / _heatSize;
    for (var c = 0; c < 4; c++) {
      int best = -128; // Minimum possible int8 value
      int bx = 0, by = 0;

      for (var y = 0; y < _heatSize; y++) {
        for (var x = 0; x < _heatSize; x++) {
          final int v = heatmaps[0][y][x][c] as int;

    if (v > best) {
      best = v;
      bx = x;
      by = y;
    }
  }
}
      corners.add(Corner(bx * scaleX, by * scaleY));
    }
    return corners;
  }

  List<List<List<List<int>>>> _preprocess(img.Image im) {
  final tensor = List.generate(
    1,
    (_) => List.generate(
      _inputSize,
      (_) => List.generate(
        _inputSize,
        (_) => List.filled(3, 0),
      ),
    ),
  );

  for (int y = 0; y < _inputSize; y++) {
    for (int x = 0; x < _inputSize; x++) {
      final p = im.getPixel(x, y);

      final r = ((p.r / 255.0 - _rm) / _rs / _qScale + _qZero)
          .round()
          .clamp(-128, 127);

      final g = ((p.g / 255.0 - _gm) / _gs / _qScale + _qZero)
          .round()
          .clamp(-128, 127);

      final b = ((p.b / 255.0 - _bm) / _bs / _qScale + _qZero)
          .round()
          .clamp(-128, 127);

      tensor[0][y][x][0] = r;
      tensor[0][y][x][1] = g;
      tensor[0][y][x][2] = b;
    }
  }

  return tensor;
}
}
