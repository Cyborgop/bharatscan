# Data Collection Protocol

Mandatory reading before capturing or sharing any document photo.

## Core rule

**Real PII never leaves the collector's laptop.** Models are trained on structure, not on real Aadhaar numbers / PAN IDs / names.

## Allowed sources

1. **Public datasets**: MIDV-500, MIDV-2020, SmartDoc-QA, DocUNet, ICDAR-MLT, IIIT-ILST
2. **Synthetic**: use `datasets/synth/generate_indic_lines.py` (trdg + Noto fonts) for OCR training lines
3. **Self-captured with explicit written consent**: only from the collector or consenting contributors, with a signed consent form (see `tools/annotation/consent_form.md`)

## Forbidden

- Scraping real ID cards from social media, Google Images, Telegram channels, etc.
- Using documents belonging to third parties without signed consent
- Uploading any real document to a cloud service (Drive, Colab, HuggingFace, Dropbox, etc.)
- Committing any raw `.jpg` / `.png` / `.pdf` of real IDs to git

## Redaction workflow

Before **any** sharing (supervisor review, GitHub issue, etc.):

```bash
python datasets/redact/redact_pii.py --input raw/ --output redacted/
```

The script blurs the 12-digit Aadhaar block, PAN ID, photo area, and signature.

## Synthetic templates for classifier

For the **document type classifier**, we do NOT need real PII. Build synthetic Aadhaar/PAN/cheque templates:

1. Download blank govt-issued template layouts (publicly available from UIDAI / Income Tax Dept press kits)
2. Fill with fake but plausible data using `trdg`
3. Apply realistic camera augmentations (perspective, noise, blur, JPEG compression)

This gives unlimited training data with zero privacy risk.

## Storage

- Raw captures: `datasets/raw/` — local only, in `.gitignore`
- Encrypted backup: 7-Zip AES-256 to external drive, not cloud
- Retention: delete raw after TFLite is trained + committed. Keep only model weights.

## DPDPA compliance

Under the Digital Personal Data Protection Act 2023, this project handles "personal data" of a "data principal". Our mitigation:

- Purpose limitation: training only
- Storage minimization: delete after use
- No transfer outside India (local storage only)
- Consent form mandatory for every contributor
