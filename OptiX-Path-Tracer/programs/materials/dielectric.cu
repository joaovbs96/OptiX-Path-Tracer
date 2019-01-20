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
rtDeclareVariable(optix::Ray, ray,   rtCurrentRay, );

// the per ray data we operate on
rtDeclareVariable(PerRayData, prd,   rtPayload, );
rtDeclareVariable(rtObject,   world, , );

// the attributes we use to communicate between intersection programs and hit program
rtDeclareVariable(Hit_Record, hit_rec, attribute hit_rec, );

// and finally - that particular material's parameters
rtBuffer< rtCallableProgramId<float3(float, float, float3)> > sample_texture; // no need to use this here
rtDeclareVariable(float, ref_idx, , );


inline __device__ bool scatter(const optix::Ray &ray_in) {
  prd.out.is_specular = true;
  prd.out.origin = hit_rec.p;
  prd.out.attenuation = vec3f(1.f);
  prd.out.normal = hit_rec.normal;
  
  vec3f outward_normal;
  float ni_over_nt;
  float cosine;
  if (dot(ray_in.direction, hit_rec.normal) > 0.f) {
    outward_normal = -1 * hit_rec.normal;
    ni_over_nt = ref_idx;
    cosine = ref_idx * dot(ray_in.direction, hit_rec.normal) / vec3f(ray_in.direction).length();
  }
  else {
    outward_normal = hit_rec.normal;
    ni_over_nt = 1.f / ref_idx;
    cosine = -dot(ray_in.direction, hit_rec.normal) / vec3f(ray_in.direction).length();
  }
  
  vec3f refracted;
  float reflect_prob;
  if (refract(ray_in.direction, outward_normal, ni_over_nt, refracted)) 
    reflect_prob = schlick(cosine, ref_idx);
  else 
    reflect_prob = 1.f;

  vec3f reflected = reflect(ray_in.direction, hit_rec.normal);
  if ((*prd.in.randState)() < reflect_prob) 
    prd.out.direction = reflected;
  else 
    prd.out.direction = refracted;
  
  return true;
}

inline __device__ float3 emitted() {
  return make_float3(0.f, 0.f, 0.f);
}

RT_PROGRAM void closest_hit() {
  prd.out.type = Dielectric;
  prd.out.emitted = emitted();
  prd.out.scatterEvent = scatter(ray) ? rayGotBounced : rayGotCancelled;
}
