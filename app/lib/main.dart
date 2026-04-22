import 'package:flutter/material.dart';
import 'screens/capture_screen.dart';

void main() => runApp(const BharatScanApp());

class BharatScanApp extends StatelessWidget {
  const BharatScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BharatScan',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFFF9933)),
        useMaterial3: true,
      ),
      home: const CaptureScreen(),
    );
  }
}
