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

// the implicit state's ray we will intersect against
rtDeclareVariable(Ray, ray, rtCurrentRay, );

// the per ray data we operate on
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );

// the attributes we use to communicate between intersection programs and hit
// program
rtDeclareVariable(Hit_Record, hit_rec, attribute hit_rec, );

// and finally - that particular material's parameters
rtBuffer<rtCallableProgramId<float3(float, float, float3)> > sample_texture;
rtDeclareVariable(float, fuzz, , );

RT_FUNCTION bool scatter(const Ray &ray_in) {
  float3 reflected = reflect(unit_vector(ray_in.direction), hit_rec.normal);
  prd.out.is_specular = true;
  prd.out.origin = hit_rec.p;
  prd.out.direction =
      reflected + fuzz * random_in_unit_sphere((*prd.in.randState));
  prd.out.attenuation =
      sample_texture[hit_rec.index](hit_rec.u, hit_rec.v, hit_rec.p);
  prd.out.normal = hit_rec.normal;
  return true;
}

RT_FUNCTION float3 emitted() { return make_float3(0.f, 0.f, 0.f); }

RT_PROGRAM void closest_hit() {
  prd.out.type = Metal;
  prd.out.emitted = emitted();
  prd.out.scatterEvent = scatter(ray) ? rayGotBounced : rayGotCancelled;
}
