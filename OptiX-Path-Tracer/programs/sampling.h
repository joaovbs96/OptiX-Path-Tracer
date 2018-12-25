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

#include "vec.h"
#include "DRand48.h"

inline __device__ vec3f random_in_unit_disk(DRand48 &rnd) {
  float a = rnd() * 2.0f * 3.1415926f;

	vec3f xy(sin(a), cos(a), 0);
	xy *= sqrt(rnd());
	
  return xy;
}

inline __device__ vec3f random_in_unit_sphere(DRand48 &rnd) {
  float z = rnd() * 2.0f - 1.0f;
	
	float t = rnd() * 2.0f * 3.1415926f;
	float r = sqrt((0.0f > (1.0f - z * z) ? 0.0f : (1.0f - z * z)));
	
	float x = r * cos(t);
	float y = r * sin(t);

	vec3f res(x, y, z);
	res *= powf(rnd(), 1.0f / 3.0f);
	
  return res;
}

inline __device__ vec3f random_cosine_direction(DRand48 &rnd){
	float r1 = rnd();
	float r2 = rnd();

	float phi = 2 * CUDART_PI_F * r1;

	float x = cos(phi) * 2 * sqrt(r2);
	float y = sin(phi) * 2 * sqrt(r2);
	float z = sqrt(1 - r2);

	return vec3f(x, y, z);
}
