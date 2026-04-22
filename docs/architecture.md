# BharatScan — Technical Architecture

Living doc. Update as decisions change.

## 1. System overview

```
Camera frame (native HD)
   │
   ▼
[1] Corner Detector  (U-Net-lite, 256×256, INT8, ~0.4 MB, ~80 ms)
   │  → 4 corner heatmaps → argmax → 4 points
   ▼
[2] Perspective Warp  (OpenCV, CPU, ~60 ms)
   │  → rectified document image (A4-ratio)
   ▼
[3] Doc Classifier   (MobileNetV3-small, 224×224, INT8, ~1.2 MB, ~40 ms)
   │  → {Aadhaar, PAN, cheque, GST invoice, marksheet, generic}
   ▼
[4] Text Detection   (DBNet-MobileNetV3, 640×640, INT8, ~1.8 MB, ~350 ms)
   │  → list of rotated polygons per text line
   ▼
[5] Text Recognition (PP-OCRv5 mobile_rec per script, 48×320 crops,
   │                  batched, INT8, ~6–8 MB per script pack, ~700 ms)
   │  → list of UTF-8 strings
   ▼
[6] Field Extractor  (regex + doc-type-aware rules, 0 MB, <5 ms)
   │  → {"aadhaar_number": "XXXX XXXX XXXX", "name": "...", ...}
   ▼
[7] PDF export / history / share (Flutter, ~0 ms model cost)
```

End-to-end target: **< 2 s on Snapdragon 6 Gen 1, < 20 MB base APK.**

## 2. Module decisions

### 2.1 Corner detector
- **Base**: custom U-Net-lite, MobileNetV3-small encoder
- **Output**: 4 × heatmap (one per corner), 64×64
- **Why**: HED is 56 MB — dead on arrival. Heatmap regression with MobileNetV3-small trains fast and quantizes cleanly
- **Fallback at ship time**: OpenCV contour-based heuristic (0 MB, 30 ms). Use when ML confidence < 0.5

### 2.2 Doc classifier
- **Base**: MobileNetV3-small @ 224×224, 6-class softmax
- **Alternative**: RepViT-M0.6 (reuse MCUDetector backbone) for +5–10% accuracy at similar size

### 2.3 Text detection
- **Base**: PP-OCRv4_mobile_det (DBNet with MobileNetV3-large backbone)
- **Why**: Paper at `2207_06966v1.pdf` and `1911_08947v2.pdf` in our notes. Segmentation-based DBNet handles multi-orientation and curved text. Lightweight and well-supported
- **Risk**: INT8 quantization around the sigmoid output is finicky. Ship FP16 if INT8 F1 drops > 3%

### 2.4 Text recognition — script families
12 languages collapse to 8 script families:

| Script | Languages | Model | Approx size |
|---|---|---|---|
| Devanagari | Hindi, Marathi | PP-OCRv5 devanagari_mobile_rec | ~8 MB |
| Latin | English | PP-OCRv5 en_mobile_rec | ~7 MB |
| Tamil | Tamil | PP-OCRv5 ta_mobile_rec | ~7.6 MB |
| Telugu | Telugu | PP-OCRv5 te_mobile_rec | ~7.6 MB |
| Bengali | Bengali, Assamese | *Needs fine-tuning from Devanagari base* | target ~8 MB |
| Gurmukhi | Punjabi | fine-tune from Devanagari | target ~7 MB |
| Perso-Arabic | Urdu | PP-OCRv5 arabic_mobile_rec (adapt) | ~7 MB |
| Dravidian-others | Kannada, Malayalam, Odia | fine-tune per script | target ~7 MB each |

**v1 ships: Latin + Devanagari + Tamil = 3 scripts bundled.** Others downloaded on demand.

### 2.5 Field extractor (regex)

| Field | Pattern |
|---|---|
| Aadhaar | `\b\d{4}\s?\d{4}\s?\d{4}\b` (with Verhoeff checksum) |
| PAN | `\b[A-Z]{5}[0-9]{4}[A-Z]\b` |
| IFSC | `\b[A-Z]{4}0[A-Z0-9]{6}\b` |
| GSTIN | `\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b` |
| Bank A/C | `\b\d{9,18}\b` (context-aware) |
| Date (DOB) | multiple formats |

Zero model cost. Add ML-based field extraction only after regex version ships.

## 3. Size budget

| Asset | Size |
|---|---|
| Flutter engine + Dart runtime | ~6 MB |
| App Dart code | ~1 MB |
| Icons, fonts (Noto Devanagari subset) | ~2 MB |
| Corner detector TFLite | 0.4 MB |
| Doc classifier TFLite | 1.2 MB |
| Text detection TFLite | 1.8 MB |
| Devanagari rec TFLite | 8 MB (largest bundled) |
| **Total base APK (install size)** | **~20.4 MB** |

Base APK compresses in Play Store to ~14 MB (download size). Tamil / Bengali / etc. fetched on first use.

## 4. Latency budget (Snapdragon 6 Gen 1, NNAPI INT8)

| Stage | Target |
|---|---|
| Corner detect | 80 ms |
| Perspective warp | 60 ms |
| Doc classify | 40 ms |
| Text detect | 350 ms |
| Text recognize (20 lines) | 700 ms |
| Regex + PDF prep | 5 ms |
| **Total** | **~1235 ms** |

Hard cap: 2000 ms. Margin: 765 ms. Measure with `scripts/measure_device_latency.sh`.

## 5. Known risks

1. **PP-OCRv5 INT8 accuracy on Indic**: undocumented. Validate in week 3 of build plan.
2. **DBNet INT8 sigmoid drift**: may need FP16 fallback (+2 MB).
3. **Aadhaar aspect ratio** differs from MIDV corpus: plan 500 self-collected samples in week 1.
4. **20 MB promise** is marketing. Clarify whether it refers to install size or Play Store download size before public commitment.
