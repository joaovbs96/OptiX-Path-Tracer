#pragma once

#include "../vec.hpp"

RT_FUNCTION float CosTheta(const float3& w) { return w.z; }

RT_FUNCTION float Cos2Theta(const float3& w) { return w.z * w.z; }

RT_FUNCTION float AbsCosTheta(const float3& w) { return abs(w.z); }

RT_FUNCTION float Sin2Theta(const float3& w) {
  return ffmax(0.f, 1.f - Cos2Theta(w));
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
  return (sinTheta == 0) ? 1.f : clamp(w.x / sinTheta, -1.f, 1.f);
}

RT_FUNCTION float SinPhi(const float3& w) {
  float sinTheta = SinTheta(w);
  return (sinTheta == 0) ? 0.f : clamp(w.y / sinTheta, -1.f, 1.f);
}

RT_FUNCTION float Cos2Phi(const float3& w) {
  float cosPhi = CosPhi(w);
  return cosPhi * cosPhi;
}

RT_FUNCTION float Sin2Phi(const float3& w) {
  float sinPhi = SinPhi(w);
  return sinPhi * sinPhi;
}

RT_FUNCTION float CosDPhi(const float3& wa, const float3& wb) {
  return clamp((wa.x * wb.x + wa.y * wb.y) / sqrtf((wa.x * wa.x + wa.y * wa.y) *
                                                   (wb.x * wb.x + wb.y * wb.y)),
               -1.f, 1.f);
}

// Returns non-normalized tangent of a hit-point
// https://computergraphics.stackexchange.com/questions/5498/compute-sphere-tangent-for-normal-mapping
RT_FUNCTION float3 Tangent(const float3& P) {
  return make_float3(-P.z, 0.f, P.x);
}