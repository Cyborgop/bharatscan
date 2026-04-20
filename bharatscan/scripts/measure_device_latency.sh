#!/usr/bin/env bash
# Measure per-module latency on a connected Android device via tflite benchmark.
# Requires adb + android-platform-tools.
set -e
DEVICE_MODELS_DIR=/data/local/tmp/bharatscan
adb shell mkdir -p $DEVICE_MODELS_DIR
for m in corner_detector_int8 doc_classifier_int8 text_det_int8 text_rec_devanagari_int8; do
  echo "=== $m ==="
  adb push app/assets/models/${m}.tflite $DEVICE_MODELS_DIR/
  adb shell /data/local/tmp/benchmark_model \
    --graph=$DEVICE_MODELS_DIR/${m}.tflite \
    --use_nnapi=true --num_threads=4 \
    --num_runs=50 --warmup_runs=10
done
