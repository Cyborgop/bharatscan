import 'package:flutter/material.dart';
class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key});
  @override
  Widget build(BuildContext c) => Scaffold(
      appBar: AppBar(title: const Text('Result')),
      body: const Center(child: Text('OCR + fields (week 3-4)')));
}
