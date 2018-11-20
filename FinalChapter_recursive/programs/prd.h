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

/*! "per ray data" (PRD) for our sample's rays. In the simple example, there is only
  one ray type, and it only ever returns one thing, which is a color (everything else
  is handled through the recursion). In addition to that return type, rays have to
  carry recursion state, which in this case are recursion depth and random number state */
struct PerRayData {
  DRand48     *randState;
  /*! recursion depth */
  int          depth;
  /*! return value of the recursive trace() call */
  vec3f        color;
};
