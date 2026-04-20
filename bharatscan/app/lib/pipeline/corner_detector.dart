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

    final input = _preprocess(resized);
    final heatmaps = List.generate(
      1,
      (_) => List.generate(
        _heatSize,
        (_) => List.generate(_heatSize, (_) => List.filled(4, 0.0)),
      ),
    );

    await runMany([input.reshape(inputShape)], {0: heatmaps});

    // Argmax per channel
    final corners = <Corner>[];
    final scaleX = frame.width / _heatSize;
    final scaleY = frame.height / _heatSize;
    for (var c = 0; c < 4; c++) {
      double best = -1;
      int bx = 0, by = 0;
      for (var y = 0; y < _heatSize; y++) {
        for (var x = 0; x < _heatSize; x++) {
          final v = heatmaps[0][y][x][c] as double;
          if (v > best) { best = v; bx = x; by = y; }
        }
      }
      corners.add(Corner(bx * scaleX, by * scaleY));
    }
    return corners;
  }

  Uint8List _preprocess(img.Image im) {
    final bytes = im.getBytes(order: img.ChannelOrder.rgb);
    final n = _inputSize * _inputSize;
    if (isQuantized) {
      final buf = Uint8List(n * 3);
      for (var i = 0; i < n; i++) {
        final j = i * 3;
        final r = (bytes[j]     / 255.0 - _rm) / _rs;
        final g = (bytes[j + 1] / 255.0 - _gm) / _gs;
        final b = (bytes[j + 2] / 255.0 - _bm) / _bs;
        buf[j]     = (r / _qScale + _qZero).round().clamp(0, 255);
        buf[j + 1] = (g / _qScale + _qZero).round().clamp(0, 255);
        buf[j + 2] = (b / _qScale + _qZero).round().clamp(0, 255);
      }
      return buf;
    }
    throw UnimplementedError('FP32 input not wired yet');
  }
}
