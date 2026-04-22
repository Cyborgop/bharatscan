// field_extractor.dart — regex-based structured field extraction.
// Zero model cost. Applies doc-type-aware rules over recognized text.

class ExtractedFields {
  final Map<String, String> fields;
  const ExtractedFields(this.fields);
}

class FieldExtractor {
  static final _aadhaar = RegExp(r'\b\d{4}\s?\d{4}\s?\d{4}\b');
  static final _pan     = RegExp(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b');
  static final _ifsc    = RegExp(r'\b[A-Z]{4}0[A-Z0-9]{6}\b');
  static final _gstin   = RegExp(r'\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b');
  static final _date    = RegExp(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b');

  static ExtractedFields extract(String docType, List<String> lines) {
    final text = lines.join('\n');
    final out = <String, String>{};

    final a = _aadhaar.firstMatch(text);
    if (a != null && _verhoeffOK(a.group(0)!.replaceAll(RegExp(r'\s'), ''))) {
      out['aadhaar_number'] = a.group(0)!;
    }
    final p = _pan.firstMatch(text);   if (p != null) out['pan'] = p.group(0)!;
    final i = _ifsc.firstMatch(text);  if (i != null) out['ifsc'] = i.group(0)!;
    final g = _gstin.firstMatch(text); if (g != null) out['gstin'] = g.group(0)!;
    final d = _date.firstMatch(text);  if (d != null) out['date'] = d.group(0)!;

    return ExtractedFields(out);
  }

  // Verhoeff checksum for Aadhaar
  static bool _verhoeffOK(String n) {
    const d = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
               [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
               [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
               [9,8,7,6,5,4,3,2,1,0]];
    const p = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
               [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
               [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]];
    if (n.length != 12 || !RegExp(r'^\d+$').hasMatch(n)) return false;
    var c = 0;
    for (var i = 0; i < n.length; i++) {
      c = d[c][p[i % 8][int.parse(n[n.length - 1 - i])]];
    }
    return c == 0;
  }
}
