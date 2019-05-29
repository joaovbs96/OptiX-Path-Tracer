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

#include "material.cuh"

////////////////////////////////////////
// --- Ideal Glass Material Model --- //
////////////////////////////////////////

// Based on:
// https://github.com/aromanro/RayTracer/blob/c8ad5de7fa91faa7e0a9de652c21284633659e2c/RayTracer/Material.cpp

// Beer-Lambert Law Theory:
// http://www.pci.tu-bs.de/aggericke/PC4/Kap_I/beerslaw.htm

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );            // ray PRD
rtDeclareVariable(rtObject, world, , );                     // scene graph
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );  // hit distance

// Intersected Geometry Parameters
rtDeclareVariable(HitRecord_Function, Get_HitRecord, , );  // HitRecord function
rtDeclareVariable(int, geo_index, attribute geo_index, );  // primitive index
rtDeclareVariable(float2, bc, attribute bc, );  // triangle barycentrics

// Material Parameters
rtDeclareVariable(Texture_Function, base_texture, , );
rtDeclareVariable(Texture_Function, extinction_texture, , );
rtDeclareVariable(float, ref_idx, , );
rtDeclareVariable(float, density, , );

RT_PROGRAM void closest_hit() {
  HitRecord rec = Get_HitRecord(geo_index, ray, t_hit, bc);
  int index = rec.index;          // texture index
  float3 P = rec.P;               // Hit Point
  float3 Wo = -rec.Wo;            // Ray view direction
  float3 N = rec.shading_normal;  // normal

  float3 base_color = base_texture(rec.u, rec.v, P, index);
  float3 absorption = make_float3(1.f);

  float ni_over_nt;
  float cosine = dot(Wo, N);

  // Ray is exiting the object
  if (cosine > 0.f) {
    N = -N;
    ni_over_nt = ref_idx;
    cosine = ref_idx * cosine / length(Wo);

    // Apply the Beer-Lambert Law
    float3 extinction = extinction_texture(rec.u, rec.v, P, index);
    // absorption = expf(-t_hit * extinction);
  }

  // Ray is entering the object
  else {
    ni_over_nt = 1.f / ref_idx;
    cosine = -cosine / length(Wo);
  }

  // Importance sample the Fresnel term
  float3 refracted;
  float reflect_prob;
  if (Refract(Wo, N, ni_over_nt, refracted))
    reflect_prob = schlick(cosine, ref_idx);
  else
    reflect_prob = 1.f;

  // Ray should be reflected...
  if (rnd(prd.seed) < reflect_prob) prd.direction = reflect(Wo, N);

  // ...or refracted
  else
    prd.direction = normalize(refracted);

  // Assign parameters to PRD
  prd.scatterEvent = rayGotBounced;
  prd.origin = P;
  prd.throughput *= (base_color * absorption);
  prd.isSpecular = true;
}
