// doc_classifier.dart
// 6-class doc type classifier — runs on the warped document image.

import 'dart:typed_data';
import 'package:image/image.dart' as img;

import '../core/tflite_base.dart';

enum DocType { aadhaar, pan}

const List<String> kDocTypeLabels = [
  'Aadhaar', 'PAN',
];

class DocClassifierService extends TfliteBase {
  static const int _inputSize = 224;

  @override
  String get assetName => 'doc_classifier.tflite';

  @override
  List<int> get inputShape => const [1, _inputSize, _inputSize, 3];

  @override
  Object buildWarmupInput() => Int8List(_inputSize * _inputSize * 3);

  Future<DocType> classify(img.Image doc) async {
    
    // TODO: preprocessing identical to training transform
    final resized = img.copyResize(
  doc,
  width: _inputSize,
  height: _inputSize,
  interpolation: img.Interpolation.linear,
);

final input = Int8List(_inputSize * _inputSize * 3);

const mean = [0.485, 0.456, 0.406];
const std = [0.229, 0.224, 0.225];

int idx = 0;

for (int y = 0; y < _inputSize; y++) {
  for (int x = 0; x < _inputSize; x++) {
    final p = resized.getPixel(x, y);

    final r = p.r / 255.0;
    final g = p.g / 255.0;
    final b = p.b / 255.0;

    final rn = (r - mean[0]) / std[0];
    final gn = (g - mean[1]) / std[1];
    final bn = (b - mean[2]) / std[2];

    input[idx++] = (rn * 128).round().clamp(-128, 127);
    input[idx++] = (gn * 128).round().clamp(-128, 127);
    input[idx++] = (bn * 128).round().clamp(-128, 127);
  }
}
    final logits = List.generate(1, (_) => List.filled(2, 0));
    await runMany([input], {0: logits});

    var best = -128; var bestIdx = 0;
    for (var i = 0; i < 2; i++) {
  if (logits[0][i] > best) {
    best = logits[0][i];
    bestIdx = i;
  }
}
    return DocType.values[bestIdx];
  }
}
