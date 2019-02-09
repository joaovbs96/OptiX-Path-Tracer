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

// TODO: use built in functions and types whenever possible

RT_FUNCTION float schlick(float cosine, float ref_idx) {
  float r0 = (1.f - ref_idx) / (1.f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.f - r0) * pow((1.f - cosine), 5.f);
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
