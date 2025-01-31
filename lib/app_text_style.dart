import 'package:flutter/material.dart';

class AppTextStyle {
  AppTextStyle._();

  static TextStyle get _base => const TextStyle();

  /// Regular
  static TextStyle w400(double fontSize) => _base.copyWith(
        fontWeight: FontWeight.w400,
        fontSize: fontSize,
      );

  /// SemiBold
  static TextStyle w600(double fontSize) => _base.copyWith(
        fontWeight: FontWeight.w600,
        fontSize: fontSize,
      );
}

extension TextStyleExtension on TextStyle {
  TextStyle get colorWhite => copyWith(color: Colors.white);
  TextStyle get colorLightGreen => copyWith(color: Colors.lightGreen);
}
