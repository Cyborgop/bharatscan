// model_downloader.dart — on-demand download of script packs.
// Base APK ships Latin + Devanagari. Tamil/Telugu/Bengali/etc. downloaded
// from a CDN on first use. Verify SHA-256 against docs/model_zoo.md.
class ModelDownloader {
  Future<String> fetch(String script) async {
    throw UnimplementedError('Implement post-v1.');
  }
}
