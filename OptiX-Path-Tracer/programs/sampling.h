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

#include "XorShift32.h"
#include "vec.h"

RT_FUNCTION float3 random_in_unit_disk(XorShift32 &rnd) {
  float a = rnd() * 2.f * PI_F;

  float3 xy = make_float3(sin(a), cos(a), 0);
  xy *= sqrt(rnd());

  return xy;
}

RT_FUNCTION float3 random_in_unit_sphere(XorShift32 &rnd) {
  float z = rnd() * 2.f - 1.f;

  float t = rnd() * 2.f * PI_F;
  float r = sqrt((0.f > (1.f - z * z) ? 0.f : (1.f - z * z)));

  float x = r * cos(t);
  float y = r * sin(t);

  float3 res = make_float3(x, y, z);
  res *= powf(rnd(), 1.f / 3.f);

  return res;
}

RT_FUNCTION float3 random_on_unit_sphere(XorShift32 &rnd) {
  float z = rnd() * 2.f - 1.f;

  float t = rnd() * 2.f * PI_F;
  float r = sqrt((0.f > (1.f - z * z) ? 0.f : (1.f - z * z)));

  float x = r * cos(t);
  float y = r * sin(t);

  float3 res = make_float3(x, y, z);
  res *= powf(rnd(), 1.f / 3.f);

  return unit_vector(res);
}

RT_FUNCTION float3 random_cosine_direction(XorShift32 &rnd) {
  float r1 = rnd();
  float r2 = rnd();

  float phi = 2 * PI_F * r1;

  float x = cos(phi) * 2.f * sqrt(r2);
  float y = sin(phi) * 2.f * sqrt(r2);
  float z = sqrt(1.f - r2);

  return make_float3(x, y, z);
}
