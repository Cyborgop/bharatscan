# Model Zoo

SHA-256 of every TFLite shipped. Update on every model release.

## Bundled in base APK (v0.1)

| Module | File | Size (MB) | SHA-256 | Metric | Value |
|---|---|---|---|---|---|
| Corner detector | `corner_detector_int8.tflite` | TBD | TBD | Corner IoU | TBD |
| Doc classifier | `doc_classifier_int8.tflite` | TBD | TBD | Top-1 acc | TBD |
| Text detection | `text_det_int8.tflite` | TBD | TBD | F-measure @ IoU 0.5 | TBD |
| Text recognition (Devanagari) | `text_rec_devanagari_int8.tflite` | TBD | TBD | CER | TBD |
| Text recognition (Latin) | `text_rec_latin_int8.tflite` | TBD | TBD | CER | TBD |

## Download-on-demand script packs

| Script | File | Size (MB) | Languages |
|---|---|---|---|
| Tamil | `text_rec_tamil_int8.tflite` | TBD | Tamil |
| Telugu | `text_rec_telugu_int8.tflite` | TBD | Telugu |
| Bengali | `text_rec_bengali_int8.tflite` | TBD | Bengali, Assamese |
| Gurmukhi | `text_rec_gurmukhi_int8.tflite` | TBD | Punjabi |
| Perso-Arabic | `text_rec_arabic_int8.tflite` | TBD | Urdu |
| Kannada | `text_rec_kannada_int8.tflite` | TBD | Kannada |
| Malayalam | `text_rec_malayalam_int8.tflite` | TBD | Malayalam |
| Odia | `text_rec_odia_int8.tflite` | TBD | Odia |
| Gujarati | `text_rec_gujarati_int8.tflite` | TBD | Gujarati |

## Versioning

Models use semver: `text_rec_devanagari_v1.2.3_int8.tflite`. The app manifest (`assets/models/manifest.json`) lists expected SHA-256 — mismatches trigger re-download.
