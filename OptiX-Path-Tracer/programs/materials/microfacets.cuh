#pragma once

#include "../math/trigonometric.cuh"
#include "../pdfs/pdf.cuh"

RT_FUNCTION bool PointingUp(const float3& v) {
  return dot(v, make_float3(0.f, 1.f, 0.f)) > 0.0f;
}

RT_FUNCTION bool SameHemiSphere(const float3& wi, const float3& wo) {
  return !(PointingUp(wi) ^ PointingUp(wo));
}

// Blinn Functions - reference:
// https://github.com/JerryCao1985/SORT/blob/547286d552dc0e555d8144fd28bfae58535ad0d8/src/bsdf/microfacet.cpp
// https://github.com/JerryCao1985/SORT/blob/547286d552dc0e555d8144fd28bfae58535ad0d8/src/bsdf/microfacet.h

RT_FUNCTION float3 Blinn_Sample(float u, float v, float nu, float nv) {
  float expU = convert_exp(nu);
  float expV = convert_exp(nv);
  float expUV = sqrt((expU + 2.0f) / (expV + 2.0f));

  float phi = 2.f * PI_F * v;
  if (expU != expV) {
    int offset[5] = {0, 1, 1, 2, 2};
    int i = v == 0.25 ? 0 : (int)(v * 4.f);
    phi = std::atan(expUV * std::tan(phi)) + offset[i] * PI_F;
  }

  float sin_phi_h = sin(phi);
  float sin_phi_h_sq = sin_phi_h * sin_phi_h;
  float alpha = expU * (1.f - sin_phi_h_sq) + expV * sin_phi_h_sq;
  float cos_theta = pow(u, 1.f / (alpha + 2.f));
  float sin_theta = sqrt(ffmax(0.f, 1.f - cos_theta * cos_theta));

  return Spherical_Vector(sin_theta, cos_theta, phi);
}

// "Returns probability of facet with given normal"
RT_FUNCTION float Blinn_Density(float3& normal, float nu, float nv) {
  float expU = convert_exp(nu);
  float expV = convert_exp(nv);
  float expUV = sqrt((expU + 2.0f) * (expV + 2.0f));

  float NoH = AbsCosTheta(normal);

  float exponent = (1 - Sin2Phi(normal)) * expU;
  exponent += Sin2Phi(normal) * expV;

  return expUV * pow(NoH, exponent) / (1.f / 2 * PI_F);
}

// "PDF of sampling a specific normal direction"
RT_FUNCTION float Blinn_PDF(float3& normal, float nu, float nv) {
  return Blinn_Density(normal, nu, nv) * AbsCosTheta(normal);
}