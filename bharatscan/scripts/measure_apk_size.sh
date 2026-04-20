#!/usr/bin/env bash
# Measure release APK sizes (per-ABI split).
set -e
cd "$(dirname "$0")/../app"
flutter build apk --release --split-per-abi
echo "--- APK sizes ---"
ls -lh build/app/outputs/flutter-apk/*.apk | awk '{print $9, $5}'
