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

#include "hitables/hitables.cuh"
#include "random.cuh"
#include "vec.hpp"

// Scatter events
typedef enum {
  rayGotBounced,    // ray could get properly bounced, and is still alive
  rayGotCancelled,  // ray could not get scattered, and should get cancelled
  rayHitLight,      // ray hit a light, and should take emission into account
  rayMissed         // ray didn't hit anything, and went into the environemnt
} ScatterEvent;

// Struct containing
struct HitRecord {
  int index;   // geometry texture index
  float3 P;    // hit point
  float t;     // distance from ray origin to hit point
  float u, v;  // texcoords
  float2 bc;   // triangle barycentric coordinates
  float3 geometric_normal;
  float3 shading_normal;
  float3 Wo;  // view direction(i.e. direction to camera)
};

// Radiance PRD containing variables that should be propagated as the ray
// scatters. It's also how the closest hit and ray gen programs communicate.
struct PerRayData {
  // data related to the current sample
  uint seed;
  float time;
  float3 throughput, radiance;

  // data related to the last hit
  ScatterEvent scatterEvent;
  bool isSpecular;

  // data related to the next ray
  float3 origin, direction;
};

// Shadow Ray PRD
struct PerRayData_Shadow {
  bool inShadow;
  float3 normal;
};
