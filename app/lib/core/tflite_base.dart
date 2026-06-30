// tflite_base.dart
// Base class for all TFLite modules in BharatScan.
// Ported from MCUDetectorService — NNAPI + IsolateInterpreter + warmup pattern.

import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

abstract class TfliteBase {
  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  bool _ready = false;
  int _warmupCount = 0;
  String _delegate = 'CPU';
  bool _isQuantized = false;

  bool get isReady => _ready && _warmupCount >= 3;
  String get delegate => _delegate;
  bool get isQuantized => _isQuantized;

  /// Asset filename under assets/models/.
  String get assetName;

  /// Input tensor shape as [1, H, W, C] or similar. Implement per-module.
  List<int> get inputShape;

  /// Override for per-module warmup dummy input construction.
  Object buildWarmupInput();

  Future<void> initialize() async {
    if (_ready) return;

    final options = InterpreterOptions()..threads = 4;
    options.useNnApiForAndroid = true;
    _delegate = 'NNAPI';

    final bytes = await rootBundle.load('assets/models/$assetName');
    final tmp = await getTemporaryDirectory();
    final file = File('${tmp.path}/$assetName');
    await file.writeAsBytes(bytes.buffer.asUint8List(), flush: true);

    try {
      _interpreter = Interpreter.fromFile(file, options: options);
    } catch (e) {
      debugPrint('[$assetName] NNAPI failed, falling back to CPU: $e');
      _delegate = 'CPU';
      _interpreter = Interpreter.fromFile(file, options: InterpreterOptions()..threads = 4);
    }

    final inType = _interpreter!.getInputTensor(0).type.toString().toLowerCase();
    _isQuantized = inType.contains('uint8');
    debugPrint('[$assetName] quantized=$_isQuantized delegate=$_delegate');

    _isolateInterpreter = await IsolateInterpreter.create(address: _interpreter!.address);

    _ready = true;

    // Warmup — 3 runs
    for (var i = 0; i < 3; i++) {
      await warmupOnce();
      _warmupCount++;
    }
  }

  Future<void> warmupOnce() async {
    // Subclasses should override with a real run. Default: no-op.
  }

  Future<void> runMany(List<Object> inputs, Map<int, Object> outputs) async {
    assert(_ready, 'Interpreter not initialized');
    await _isolateInterpreter!.runForMultipleInputs(inputs, outputs);
  }

  Future<void> dispose() async {
    await _isolateInterpreter?.close();
    _interpreter?.close();
    _interpreter = null;
    _isolateInterpreter = null;
    _ready = false;
    _warmupCount = 0;
  }
}
