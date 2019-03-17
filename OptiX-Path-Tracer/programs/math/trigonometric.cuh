#pragma once

#include "../vec.hpp"

RT_FUNCTION float3 MatrixMultiply(const float3& vec, const Matrix3x3& mat) {
  return make_float3(vec.x * mat.getRow(0).x + vec.y * mat.getRow(1).x +
                         vec.z * mat.getRow(2).x,
                     vec.x * mat.getRow(0).y + vec.y * mat.getRow(1).y +
                         vec.z * mat.getRow(2).y,
                     vec.x * mat.getRow(0).z + vec.y * mat.getRow(1).z +
                         vec.z * mat.getRow(2).z);
}

RT_FUNCTION float CosTheta(const float3& w) { return w.y; }

RT_FUNCTION float Cos2Theta(const float3& w) { return w.y * w.y; }

RT_FUNCTION float AbsCosTheta(const float3& w) { return abs(CosTheta(w)); }

RT_FUNCTION float Sin2Theta(const float3& w) {
  return ffmax(0.0f, 1.0f - Cos2Theta(w));
}

RT_FUNCTION float SinTheta(const float3& w) { return sqrtf(Sin2Theta(w)); }

RT_FUNCTION float TanTheta(const float3& w) {
  return SinTheta(w) / CosTheta(w);
}

RT_FUNCTION float Tan2Theta(const float3& w) {
  return Sin2Theta(w) / Cos2Theta(w);
}

RT_FUNCTION float CosPhi(const float3& w) {
  float sinTheta = SinTheta(w);
  return (sinTheta == 0) ? 1.0f : clamp(w.x / sinTheta, -1.0f, 1.0f);
}

RT_FUNCTION float SinPhi(const float3& w) {
  float sinTheta = SinTheta(w);
  return (sinTheta == 0) ? 1.0f : clamp(w.z / sinTheta, -1.0f, 1.0f);
}

RT_FUNCTION float Cos2Phi(const float3& w) {
  float cosPhi = CosPhi(w);
  return cosPhi * cosPhi;
}

RT_FUNCTION float Sin2Phi(const float3& w) {
  float sinPhi = SinPhi(w);
  return sinPhi * sinPhi;
}