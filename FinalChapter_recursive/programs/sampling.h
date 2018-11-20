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

inline __device__ vec3f random_in_unit_disk(DRand48 &local_rand_state) {
  vec3f p;
  do {
    p = 2.0f*vec3f(local_rand_state(), local_rand_state(), 0) - vec3f(1, 1, 0);
  } while (dot(p, p) >= 1.0f);
  return p;
}


#define RANDVEC3F vec3f(rnd(),rnd(),rnd())

inline __device__ vec3f random_in_unit_sphere(DRand48 &rnd) {
  vec3f p;
  do {
    p = 2.0f*RANDVEC3F - vec3f(1, 1, 1);
  } while (p.squared_length() >= 1.0f);
  return p;
}

