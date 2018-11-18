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

#include "material.h"

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );

inline __device__ vec3f missColor(const optix::Ray &ray)
{
  const vec3f unit_direction = normalize(ray.direction);
  const float t = 0.5f*(unit_direction.y + 1.0f);
  const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  return c;
}

RT_PROGRAM void miss_program()
{
  prd.color  = missColor(ray);
}
