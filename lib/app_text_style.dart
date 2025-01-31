import 'package:flutter/material.dart';

class AppTextStyle {
  AppTextStyle._();

  static TextStyle get _base => const TextStyle();

  /// Regular
  static TextStyle w400(double fontSize) => _base.copyWith(
        fontWeight: FontWeight.w400,
        fontSize: fontSize,
      );

  /// Bold
  static TextStyle w700(double fontSize) => _base.copyWith(
        fontWeight: FontWeight.w700,
        fontSize: fontSize,
      );
}

extension TextStyleExtension on TextStyle {
  TextStyle get colorWhite => copyWith(color: Colors.white);
  TextStyle get colorGreen => copyWith(color: Colors.green);
}
