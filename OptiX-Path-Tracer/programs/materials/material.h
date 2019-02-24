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

#include "../XorShift32.h"
#include "../pdfs/pdf.h"
#include "../prd.h"
#include "../sampling.h"
#include "../trigonometric.h"

RT_FUNCTION float schlick(float cosine, float ref_idx) {
  float r0 = (1.f - ref_idx) / (1.f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.f - r0) * pow((1.f - cosine), 5.f);
}

RT_FUNCTION float3 schlick(float3 r0, float radians) {
  float exponential = powf(1.f - radians, 5.f);
  return r0 + (make_float3(1.f) - r0) * exponential;
}

RT_FUNCTION float SchlickR0FromRelativeIOR(float eta) {
  // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
  return ((eta - 1.f) * (eta - 1.f)) / ((eta + 1.f) * (eta + 1.f));
}

RT_FUNCTION bool refract(const float3& v, const float3& n, float ni_over_nt,
                         float3& refracted) {
  float3 uv = unit_vector(v);
  float dt = dot(uv, n);
  float discriminant = 1.f - ni_over_nt * ni_over_nt * (1.f - dt * dt);

  if (discriminant > 0.f) {
    refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
    return true;
  } else
    return false;
}

RT_FUNCTION float Fresnel_Dielectric(float cosThetaI, float ni, float nt) {
  // Copied from PBRT. This function calculates the full Fresnel term for a
  // dielectric material. See Sebastion Legarde's link above for details.

  cosThetaI = Clamp(cosThetaI, -1.f, 1.f);

  // Swap index of refraction if this is coming from inside the surface
  if (cosThetaI < 0.f) {
    float temp = ni;
    ni = nt;
    nt = temp;

    cosThetaI = -cosThetaI;
  }

  float sinThetaI = sqrtf(ffmax(0.f, 1.f - cosThetaI * cosThetaI));
  float sinThetaT = ni / nt * sinThetaI;

  // Check for total internal reflection
  if (sinThetaT >= 1) return 1.f;

  float cosThetaT = sqrtf(ffmax(0.f, 1.f - sinThetaT * sinThetaT));

  float rParallel = ((nt * cosThetaI) - (ni * cosThetaT)) /
                    ((nt * cosThetaI) + (ni * cosThetaT));
  float rPerpendicuar = ((ni * cosThetaI) - (nt * cosThetaT)) /
                        ((ni * cosThetaI) + (nt * cosThetaT));
  return (rParallel * rParallel + rPerpendicuar * rPerpendicuar) / 2;
}

RT_FUNCTION float3 CalculateExtinction(float3 apparantColor,
                                       float scatterDistance) {
  float3 a = apparantColor;
  float3 s = make_float3(1.9f) - a +
             3.5f * (a - make_float3(0.8f)) * (a - make_float3(0.8f));

  return 1.0f / (s * scatterDistance);
}

RT_FUNCTION bool Transmit(float3 wm, float3 wi, float n, float3& wo) {
  float c = dot(wi, wm);
  if (c < 0.f) {
    c = -c;
    wm = -wm;
  }

  float root = 1.f - n * n * (1.f - c * c);
  if (root <= 0.f) return false;

  wo = (n * c - sqrt(root)) * wm - n * wi;
  return true;
}