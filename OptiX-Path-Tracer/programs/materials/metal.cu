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
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

// and finally - that particular material's parameters
rtBuffer<rtCallableProgramId<float3(float, float, float3)> > sample_texture;
rtDeclareVariable(float, fuzz, , );  // how 'rough'/fuzzy the metal is

RT_PROGRAM void closest_hit() {
  prd.matType = Metal_Material;
  prd.isSpecular = true;
  prd.scatterEvent = rayGotBounced;

  prd.origin = hit_rec.p;
  prd.normal = hit_rec.normal;

  float3 reflected = reflect(unit_vector(ray.direction), prd.normal);
  prd.direction = reflected + fuzz * random_in_unit_sphere(prd.seed);

  int index = hit_rec.index;
  prd.emitted = make_float3(0.f);
  prd.attenuation = sample_texture[index](hit_rec.u, hit_rec.v, hit_rec.p);
}

RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  return make_float3(1.f);
}

RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) { return 1.f; }

RT_CALLABLE_PROGRAM float BRDF_Evaluate(PDFParams &pdf) { return 1.f; }
