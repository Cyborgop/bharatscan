// doc_classifier.dart
// 6-class doc type classifier — runs on the warped document image.

import 'dart:typed_data';
import 'package:image/image.dart' as img;

import '../core/tflite_base.dart';

enum DocType { aadhaar, pan, cheque, gstInvoice, marksheet, generic }

const List<String> kDocTypeLabels = [
  'Aadhaar', 'PAN', 'Cheque', 'GST Invoice', 'Marksheet', 'Generic',
];

class DocClassifierService extends TfliteBase {
  static const int _inputSize = 224;

  @override
  String get assetName => 'doc_classifier_int8.tflite';

  @override
  List<int> get inputShape => const [1, _inputSize, _inputSize, 3];

  @override
  Object buildWarmupInput() => Uint8List(_inputSize * _inputSize * 3);

  Future<DocType> classify(img.Image doc) async {
    final resized = img.copyResize(doc,
        width: _inputSize, height: _inputSize,
        interpolation: img.Interpolation.linear);
    // TODO: preprocessing identical to training transform
    final input = resized.getBytes(order: img.ChannelOrder.rgb);
    final logits = List.generate(1, (_) => List.filled(6, 0.0));
    await runMany([input.reshape(inputShape)], {0: logits});

    var best = -1e9; var bestIdx = 0;
    for (var i = 0; i < 6; i++) {
      if (logits[0][i] > best) { best = logits[0][i]; bestIdx = i; }
    }
    return DocType.values[bestIdx];
  }
}
