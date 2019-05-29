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

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData, prd, rtPayload, );

// Texture Samplers
// TODO: remove one of these
rtDeclareVariable(Texture_Function, sample_color1, , );
rtDeclareVariable(Texture_Function, sample_color2, , );
rtDeclareVariable(Texture_Function, sample_texture, , );

// Gradient Color Background
RT_PROGRAM void gradient_color() {
  const float3 unit_direction = normalize(ray.direction);
  const float t = 0.5f * (unit_direction.y + 1.f);

  // make gradient color
  float3 c = (1.f - t) * sample_color1(0, 0, make_float3(0.f), 0);
  c += t * sample_color2(0, 0, make_float3(0.f), 0);

  prd.throughput *= c;
  prd.scatterEvent = rayMissed;
}

// Constant Color Background
RT_PROGRAM void constant_color() {
  prd.throughput *= sample_texture(0, 0, make_float3(0.f), 0);
  prd.scatterEvent = rayMissed;
}

// RGB Image Background
// from OptiX Quick-start Tutorial
// https://docs.nvidia.com/gameworks/content/gameworkslibrary/optix/optix_quickstart.htm
RT_PROGRAM void image_background() {
  float theta = atan2f(ray.direction.x, ray.direction.z);
  float phi = M_PIf * 0.5f - acosf(ray.direction.y);
  float u = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v = 0.5f * (1.f + sinf(phi));

  prd.throughput *= sample_texture(u, v, make_float3(0.f), 0);
  prd.scatterEvent = rayMissed;
}

// HDRi Environmental Mapping
rtDeclareVariable(int, isSpherical, , );

RT_PROGRAM void environmental_mapping() {
  float u, v;

  // spherical HDRI mapping
  if (isSpherical) {
    // https://www.gamedev.net/forums/topic/637220-equirectangular-environment-map/
    float r = length(ray.direction);
    float lon = atan2(ray.direction.z, ray.direction.x);
    float lat = acos(ray.direction.y / r);

    float2 rads = make_float2(1.f / (PI_F * 2.f), 1.f / PI_F);
    u = lon * rads.x;
    v = lat * rads.y;
  }

  // cylindrical HDRI mapping
  else {
    // Y is up, swap x for y and z for x
    float theta = atan2f(ray.direction.x, ray.direction.z);

    // wrap around full circle if negative
    theta = theta < 0.f ? theta + (2.f * PI_F) : theta;
    float phi = acosf(ray.direction.y);

    // map theta and phi to u and v texturecoordinates in [0,1] x [0,1] range
    u = 1.f - (theta / (2.f * PI_F));
    v = phi / PI_F;
  }

  prd.throughput *= 2.f * sample_texture(u, v, make_float3(0.f), 0);
  prd.scatterEvent = rayMissed;
}