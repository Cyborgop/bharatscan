// perspective_warp.dart
// Pure-Dart perspective correction using 4 detected corners.
// For production: consider native OpenCV via ffi — 10× faster on large images.

import 'package:image/image.dart' as img;

import 'corner_detector.dart';

class PerspectiveWarp {
  /// Warp the document quadrilateral to an A4-ratio rectangle.
  static img.Image warp(img.Image src, List<Corner> corners,
      {int outW = 1240, int outH = 1754}) {
    assert(corners.length == 4);

    // image ^4.x has a perspective transform helper
    final tl = corners[0], tr = corners[1], br = corners[2], bl = corners[3];

    return img.copyRectify(
      src,
      topLeft: img.Point(tl.x, tl.y),
      topRight: img.Point(tr.x, tr.y),
      bottomRight: img.Point(br.x, br.y),
      bottomLeft: img.Point(bl.x, bl.y),
      toImage: img.Image(width: outW, height: outH),
    );
  }
}
