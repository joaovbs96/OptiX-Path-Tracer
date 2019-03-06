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

#include "materials/material.cuh"

// the implicit state's ray we will intersect against
rtDeclareVariable(Ray, ray, rtCurrentRay, );
// the per ray data we operate on
rtDeclareVariable(PerRayData, prd, rtPayload, );

RT_PROGRAM void sky() {
  const float3 unit_direction = normalize(ray.direction);
  const float t = 0.5f * (unit_direction.y + 1.f);
  const float3 c =
      (1.f - t) * make_float3(1.f) + t * make_float3(0.5f, 0.7f, 1.f);
  prd.attenuation = c;
  prd.scatterEvent = rayMissed;
}

RT_PROGRAM void dark() {
  prd.attenuation = make_float3(0.f);
  prd.scatterEvent = rayMissed;
}

rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  sample_texture, , );

RT_PROGRAM void box() {
  prd.attenuation = make_float3(0.f);
  prd.scatterEvent = rayMissed;
}

// rgbe image background
RT_PROGRAM void img_background() {
  float theta = atan2f(ray.direction.x, ray.direction.z);
  float phi = M_PIf * 0.5f - acosf(ray.direction.y);
  float u = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v = 0.5f * (1.f + sinf(phi));

  prd.attenuation = sample_texture(u, v, make_float3(0.f), 0);
  prd.scatterEvent = rayMissed;
}

rtDeclareVariable(int, isSpherical, , );

// HDRi environmental mapping
RT_PROGRAM void environmental_mapping() {
  float u, v;

  if (isSpherical) {
    // https://www.gamedev.net/forums/topic/637220-equirectangular-environment-map/
    float r = length(ray.direction);
    float lon = atan2(ray.direction.z, ray.direction.x);
    float lat = acos(ray.direction.y / r);

    float2 rads = make_float2(1.f / (PI_F * 2.f), 1.f / PI_F);
    u = lon * rads.x;
    v = lat * rads.y;

  } else {  // cylindrical HDRI mapping
    // Y is up, swap x for y and z for x
    float theta = atan2f(ray.direction.x, ray.direction.z);
    // wrap around full circle if negative
    theta = theta < 0.f ? theta + (2.f * PI_F) : theta;
    float phi = acosf(ray.direction.y);

    // map theta and phi to u and v texturecoordinates in [0,1] x [0,1] range
    u = 1.f - (theta / (2.f * PI_F));  // +offsetY;
    v = phi / PI_F;
  }

  prd.attenuation = 2.f * sample_texture(u, v, make_float3(0.f), 0);
  prd.scatterEvent = rayMissed;
}