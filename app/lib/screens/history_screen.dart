import 'package:flutter/material.dart';
class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});
  @override
  Widget build(BuildContext c) => Scaffold(
      appBar: AppBar(title: const Text('History')),
      body: const Center(child: Text('sqflite-backed history (week 4)')));
}
