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

// disney principled material
struct Disney_Parameters {
  float3 baseColor;
  float3 transmittanceColor;
  float sheen;
  float sheenTint;
  float clearcoat;
  float clearcoatGloss;
  float metallic;
  float specTrans;
  float diffTrans;
  float flatness;
  float anisotropic;
  float relativeIOR;
  float specularTint;
  float roughness;
  float scatterDistance;

  float ior;
};

// TODO: clean up material parameters
// Parameters used in some BRDF callable programs
struct BRDFParameters {
  BRDFType type;
  int index;
  float u, v;
  float3 attenuation;

  // sub-structs to hold material specific params
  // anisotropic material
  struct {
    float3 diffuse_color;
    float3 specular_color;
    float nu, nv;
  } anisotropic;

  // oren-nayar material
  struct {
    float rA, rB;
  } orenNayar;

  Disney_Parameters disney;
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
  float3 view_direction;    // Wo, direction of the previous ray
  float3 origin;            // P, hit point, origin of the next ray
  float3 direction;         // Wi, direction of the next ray
  float3 geometric_normal;  // Ng
  float3 shading_normal;    // Ns
  float3 emitted;
  float3 attenuation;
  float3 throughput;
  bool isSpecular;
  BRDFType matType;
  BRDFParameters matParams;  // TODO: remove this from here
};

struct PerRayData_Shadow {
  bool inShadow;
};
