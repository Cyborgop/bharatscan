# BharatScan

Offline-first document scanner for India. CamScanner-class features, privacy-first, runs entirely on-device, APK under 20 MB, works on 2 GB RAM Android phones.

## What it does

1. Capture a document with the camera
2. Detect the 4 corners and perspective-correct it
3. Classify the document type (Aadhaar / PAN / cheque / GST invoice / marksheet / generic)
4. Detect and recognize text in Hindi, English, and 10 other Indian languages
5. Extract structured fields (Aadhaar number, PAN, IFSC, etc.) using regex
6. Export to PDF, save to encrypted local history, share

Nothing leaves the device. No cloud. No internet needed.

## Core pillars

- **100% on-device inference** — no upload, no cloud
- **APK under 20 MB** — fits on any Android phone
- **Runs on 2 GB RAM** — targets Snapdragon 4/6/7-series
- **12 Indian scripts** — Devanagari (Hindi, Marathi), Bengali (Bengali, Assamese), Tamil, Telugu, Kannada, Malayalam, Gujarati, Gurmukhi (Punjabi), Odia, Perso-Arabic (Urdu), Latin (English)
- **Document presets** — smart defaults for Aadhaar, PAN, cheque, GST invoice, marksheet
- **Voice-guided capture** — in regional languages, for first-time users
- **DPDPA compliant** — made in India, for India

## Tech stack

| Layer | Choice |
|---|---|
| Training | PyTorch, RTX 2080 Ti (IIT KGP) |
| Quantization | ONNX → onnx2tf → TFLite INT8 |
| Mobile runtime | Flutter + `tflite_flutter 0.12.1`, NNAPI delegate, `IsolateInterpreter` |
| Storage | `sqflite` (encrypted) |
| PDF export | `pdf` Flutter package |
| Target | Android 10+, Snapdragon 4/6/7-series |

## Pipeline modules

1. Document edge detection + perspective correction
2. Document type classifier
3. Text detection (DBNet-mobile)
4. Text recognition (PP-OCRv5 mobile, per script family)
5. Structured field extraction (regex)
6. PDF export, local history, share sheet

## Repository layout

```
bharatscan/
├── models/           # PyTorch training code (one subdir per module)
├── quantization/     # ONNX → TFLite INT8 pipeline
├── datasets/         # Download + synthesis scripts. NEVER commit raw data.
├── evaluation/       # CER/WER, end-to-end tests
├── app/              # Flutter app
├── scripts/          # APK size + latency measurement
└── docs/             # Architecture, latency budget, model zoo, data protocol
```

## Quick start

```bash
# Training side (IIT KGP server)
conda create -n bharatscan python=3.10 -y
conda activate bharatscan
pip install -r requirements.txt

# Train corner detector
python models/corner_detector/train.py --config models/corner_detector/config.yaml

# Quantize
python quantization/onnx_to_tflite.py --module corner_detector

# App side
cd app
flutter pub get
flutter run --release
```

## Status

Pre-launch. Research + architecture phase. See `docs/architecture.md` for the full technical plan.

## License

See `LICENSE`.

## Author

Subhadeep Mondal, IIT Kharagpur.

