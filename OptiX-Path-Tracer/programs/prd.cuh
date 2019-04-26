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

#include "random.cuh"
#include "vec.hpp"

typedef enum {
  rayGotBounced,    // ray could get properly bounced, and is still alive
  rayGotCancelled,  // ray could not get scattered, and should get cancelled
  rayHitLight,      // ray hit a light, and should take emission into account
  rayMissed,        // ray didn't hit anything, and went into the environemnt
} ScatterEvent;

struct HitRecord {
  int index;
  float distance;
  float u;
  float v;
  float3 geometric_normal;
  float3 shading_normal;
  float3 front_P;
  float3 back_P;
  float3 p;
  float3 view_direction;
};

/*! "per ray data" (PRD) for our sample's rays. In the simple example, there is
  only one ray type, and it only ever returns one thing, which is a color
  (everything else is handled through the recursion). In addition to that return
  type, rays have to carry recursion state, which in this case are recursion
  depth and random number state */
struct PerRayData {
  uint seed;
  float time;

  ScatterEvent scatterEvent;
  float3 origin;
  float3 direction;
  float3 throughput;
  float3 radiance;
  bool isSpecular;
};

struct PerRayData_Shadow {
  bool inShadow;
};
