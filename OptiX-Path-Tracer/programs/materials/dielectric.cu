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

// TODO: update dielectric to make use of beer lambert's law
rtBuffer<rtCallableProgramId<float3(float, float, float3)> > sample_texture;
rtDeclareVariable(float, ref_idx, , );

RT_PROGRAM void closest_hit() {
  prd.matType = Dielectric_Material;
  prd.isSpecular = true;
  prd.scatterEvent = rayGotBounced;

  prd.origin = hit_rec.p;
  prd.normal = hit_rec.normal;

  float3 outward_normal;
  float ni_over_nt;
  float cosine;
  if (dot(ray.direction, prd.normal) > 0.f) {
    outward_normal = -1 * prd.normal;
    ni_over_nt = ref_idx;
    cosine = ref_idx * dot(ray.direction, prd.normal) / length(ray.direction);
  } else {
    outward_normal = prd.normal;
    ni_over_nt = 1.f / ref_idx;
    cosine = -dot(ray.direction, prd.normal) / length(ray.direction);
  }

  float3 refracted;
  float reflect_prob;
  if (refract(ray.direction, outward_normal, ni_over_nt, refracted))
    reflect_prob = schlick(cosine, ref_idx);
  else
    reflect_prob = 1.f;

  float3 reflected = reflect(ray.direction, prd.normal);
  if ((*prd.randState)() < reflect_prob)
    prd.direction = reflected;
  else
    prd.direction = refracted;

  prd.emitted = make_float3(0.f);
  prd.attenuation = make_float3(1.f);
}

RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, XorShift32 &rnd) {
  return make_float3(1.f);
}

RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) { return 1.f; }

RT_CALLABLE_PROGRAM float BRDF_Evaluate(PDFParams &pdf) { return 1.f; }
