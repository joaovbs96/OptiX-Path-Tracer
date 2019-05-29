// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "../math/trigonometric.cuh"
#include "../prd.cuh"
#include "../sampling.cuh"
#include "../vec.hpp"

// Typedef of Texture callable program calls
typedef rtCallableProgramId<float3(float, float, float3, int)> Texture_Function;

// Typedef of geometry parameters callable program calls
typedef rtCallableProgramX<HitRecord(int, Ray, float, float2)> HitRecord_Function;

RT_FUNCTION float schlick(float cosine, float ref_idx) {
  float r0 = (1.f - ref_idx) / (1.f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.f - r0) * powf((1.f - cosine), 5.f);
}

RT_FUNCTION float3 schlick(float3 r0, float cosine) {
  float exponential = powf(1.f - cosine, 5.f);
  return r0 + (make_float3(1.f) - r0) * exponential;
}

RT_FUNCTION float SchlickWeight(float cos) {
  return powf(saturate(1.0f - cos), 5.0f);
}

RT_FUNCTION float SchlickR0FromRelativeIOR(float eta) {
  // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
  return ((eta - 1.f) * (eta - 1.f)) / ((eta + 1.f) * (eta + 1.f));
}

RT_FUNCTION bool Refract(const float3& v,   // origin
                         const float3& n,   // normal
                         float ni_over_nt,  // ni over nt
                         float3& refracted) {
  float3 uv = normalize(v);
  float dt = dot(uv, n);
  float discriminant = 1.f - ni_over_nt * ni_over_nt * (1.f - dt * dt);

  if (discriminant > 0.f) {
    refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
    return true;
  } else
    return false;
}

RT_FUNCTION float FrDielectric(float _cosThetaI, float _etaI, float _etaT) {
  float etaI = _etaI, etaT = _etaT;
  float cosThetaI = clamp(_cosThetaI, -1.f, 1.f);
  // Potentially swap indices of refraction
  bool entering = cosThetaI > 0.f;
  if (!entering) {
    // float temp = etaI;
    etaI = etaT;
    etaT = etaI;
    cosThetaI = fabsf(cosThetaI);
  }

  // Compute _cosThetaT_ using Snell's law
  float sinThetaI = sqrtf(fmaxf(0.f, 1.f - cosThetaI * cosThetaI));
  float sinThetaT = etaI / etaT * sinThetaI;

  // Handle total internal reflection
  if (sinThetaT >= 1) return 1.f;
  float cosThetaT = sqrtf(fmaxf(0.f, 1.f - sinThetaT * sinThetaT));
  float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                ((etaT * cosThetaI) + (etaI * cosThetaT));
  float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                ((etaI * cosThetaI) + (etaT * cosThetaT));
  return (Rparl * Rparl + Rperp * Rperp) / 2;
}