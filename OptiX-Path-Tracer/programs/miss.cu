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

#include "materials/material.h"

// the implicit state's ray we will intersect against
rtDeclareVariable(Ray, ray, rtCurrentRay, );
// the per ray data we operate on
rtDeclareVariable(PerRayData, prd, rtPayload, );

RT_PROGRAM void sky() {
  const float3 unit_direction = normalize(ray.direction);
  const float t = 0.5f * (unit_direction.y + 1.f);
  const float3 c =
      (1.f - t) * make_float3(1.f) + t * make_float3(0.5f, 0.7f, 1.f);
  prd.out.attenuation = prd.out.emitted = c;
  prd.out.scatterEvent = rayDidntHitAnything;
}

RT_PROGRAM void dark() {
  prd.out.attenuation = prd.out.emitted = make_float3(0.f);
  prd.out.scatterEvent = rayDidntHitAnything;
}

rtBuffer<rtCallableProgramId<float3(float, float, float3)> > sample_texture;

RT_PROGRAM void box() {
  prd.out.attenuation = prd.out.emitted = make_float3(0.f);
  prd.out.scatterEvent = rayDidntHitAnything;
}

RT_PROGRAM void environmental_mapping() {
  float theta = atan2f(ray.direction.x, ray.direction.z);
  float phi = M_PIf * 0.5f - acosf(ray.direction.y);
  float u = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v = 0.5f * (1.f + sinf(phi));

  prd.out.attenuation = prd.out.emitted =
      sample_texture[0](u, v, make_float3(0.f));
  prd.out.scatterEvent = rayDidntHitAnything;
}