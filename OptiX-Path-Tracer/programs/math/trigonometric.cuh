#pragma once

#include "../vec.hpp"

RT_FUNCTION float CosTheta(const float3& w) { return w.y; }

RT_FUNCTION float Cos2Theta(const float3& w) { return w.y * w.y; }

RT_FUNCTION float AbsCosTheta(const float3& w) { return abs(w.y); }

RT_FUNCTION float Sin2Theta(const float3& w) {
  return fmaxf(0.f, 1.f - Cos2Theta(w));
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
  return (sinTheta == 0) ? 0.f : clamp(w.z / sinTheta, -1.f, 1.f);
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

RT_FUNCTION float Spherical_Theta(const float3& v) {
  return acosf(clamp(v.y, -1.f, 1.f));
}

RT_FUNCTION float Spherical_Phi(const float3& v) {
  float p = atan2f(v.z, v.x);
  return (p < 0.f) ? p + 2.f * PI_F : p;
}

RT_FUNCTION bool Same_Hemisphere(const float3 a, const float3 b) {
  return a.z * b.z > 0.f;
}

// Returns non-normalized tangent of a hit-point
// https://computergraphics.stackexchange.com/questions/5498/compute-sphere-tangent-for-normal-mapping
RT_FUNCTION float3 Tangent(const float3& P) {
  return make_float3(-P.z, 0.f, P.x);
}

RT_FUNCTION void Make_Orthonormals(const float3 N, float3& a, float3& b) {
  if (N.x != N.y || N.x != N.z)
    a = make_float3(N.z - N.y, N.x - N.z, N.y - N.x);  //(1,1,1)x N
  else
    a = make_float3(N.z - N.y, N.x + N.z, -N.y - N.x);  //(-1,1,1)x N

  a = normalize(a);
  b = cross(N, a);
}

/* return an orthogonal tangent and bitangent given a normal and tangent that
 * may not be exactly orthogonal */
RT_FUNCTION void Make_Orthonormals_Tangent(const float3 N, const float3 T,
                                           float3& a, float3& b) {
  b = normalize(cross(N, T));
  a = cross(b, N);
}

// transform vector from world coordinate to shading coordinate
RT_FUNCTION float3 WorldToLocal(const Onb& uvw, const float3& P) {
  return make_float3( dot(uvw.m_tangent, P), dot(uvw.m_normal, P), dot(uvw.m_binormal, P) );
}

// transform vector from shading coordinate to world coordinate
RT_FUNCTION float3 LocalToWorld(const Onb& uvw, const float3& P) {
  return make_float3( 	P.x * uvw.m_tangent.x + P.y * uvw.m_normal.x + P.z * uvw.m_binormal.x ,
                        P.x * uvw.m_tangent.y + P.y * uvw.m_normal.y + P.z * uvw.m_binormal.y ,
                        P.x * uvw.m_tangent.z + P.y * uvw.m_normal.z + P.z * uvw.m_binormal.z );
}

RT_FUNCTION float3 Spherical_Vector(float sintheta, float costheta, float phi) {
  return make_float3(sintheta * cosf(phi), costheta, sintheta * sinf(phi));
}

RT_FUNCTION float3 Spherical_Vector(float theta, float phi) {
  return make_float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}