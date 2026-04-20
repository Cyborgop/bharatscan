// text_recognizer.dart — PP-OCRv5 mobile_rec, batched over text-line crops.
// TODO(week 3): implement preprocessing (48×320), CTC decode, batched runs.

import 'dart:typed_data';
import 'package:image/image.dart' as img;
import '../core/tflite_base.dart';
import 'text_detector.dart';

class RecognizedLine {
  final String text;
  final double score;
  final TextPolygon polygon;
  const RecognizedLine(this.text, this.score, this.polygon);
}

class TextRecognizerService extends TfliteBase {
  final String script; // 'devanagari', 'latin', etc.
  TextRecognizerService(this.script);

  @override
  String get assetName => 'text_rec_${script}_int8.tflite';
  @override
  List<int> get inputShape => const [1, 48, 320, 3];
  @override
  Object buildWarmupInput() => Uint8List(48 * 320 * 3);

  Future<List<RecognizedLine>> recognize(
      img.Image doc, List<TextPolygon> polys) async {
    throw UnimplementedError('Implement in week 3.');
  }
}
