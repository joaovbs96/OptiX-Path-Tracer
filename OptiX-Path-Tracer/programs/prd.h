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

typedef enum {
  Lambertian,
  Diffuse_Light,
  Metal,
  Dielectric,
  Isotropic
} Material_Type;

typedef enum {
  /*! ray could get properly bounced, and is still alive */
  rayGotBounced,
  /*! ray could not get scattered, and should get cancelled */
  rayGotCancelled,
  /*! ray didn't hit anything, and went into the environemnt */
  rayDidntHitAnything
} ScatterEvent;

struct Hit_Record {
  vec3f normal;
  vec3f p;
  float distance;
  float u;
  float v;
};

/*! "per ray data" (PRD) for our sample's rays. In the simple example, there is only
  one ray type, and it only ever returns one thing, which is a color (everything else
  is handled through the recursion). In addition to that return type, rays have to
  carry recursion state, which in this case are recursion depth and random number state */
struct PerRayData {
  struct {
    DRand48 *randState;
    float time;
  } in;
  struct {
    ScatterEvent  scatterEvent;
    vec3f         origin;
    vec3f         direction;
    vec3f         normal;
    vec3f         emitted;
    vec3f         attenuation;
    bool          is_specular;
    Material_Type type;
  } out;
};
