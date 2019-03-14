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

// TODO: clean up material parameters
// Parameters used in some BRDF callable programs
struct MaterialParameters {
  MaterialType type;
  int index;
  float u, v;
  float3 attenuation;

  // anisotropic material specific parameters
  struct {
    float3 diffuse_color;
    float3 specular_color;
    float nu, nv;
  } anisotropic;
};

typedef enum {
  /*! ray could get properly bounced, and is still alive */
  rayGotBounced,
  /*! ray could not get scattered, and should get cancelled */
  rayGotCancelled,
  /*! ray didn't hit anything, and went into the environemnt */
  rayMissed,
} ScatterEvent;

struct HitRecord {
  int index;
  float distance;
  float u;
  float v;
  float3 geometric_normal;
  float3 shading_normal;
  float3 p;
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
  float3 geometric_normal;
  float3 shading_normal;
  float3 emitted;
  float3 attenuation;
  float3 throughput;
  bool isSpecular;
  MaterialType matType;
  MaterialParameters matParams;
};

struct PerRayData_Shadow {
  bool inShadow;
};
