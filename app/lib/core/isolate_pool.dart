// isolate_pool.dart
// Thin wrapper around Isolate for running CPU-heavy ops (text recognition
// batching, PDF rendering) off the UI thread.

import 'dart:async';
import 'dart:isolate';

typedef IsolateTask<R> = FutureOr<R> Function();

class IsolatePool {
  /// Run a pure-Dart function in a background isolate.
  /// For TFLite ops, prefer IsolateInterpreter instead.
  static Future<R> run<R>(FutureOr<R> Function() task) {
    return Isolate.run(task);
  }
}
