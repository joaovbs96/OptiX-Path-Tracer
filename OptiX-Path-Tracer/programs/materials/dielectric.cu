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

// the implicit state's ray we will intersect against
rtDeclareVariable(Ray, ray, rtCurrentRay, );

// the per ray data we operate on
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );

// the attributes we use to communicate between intersection programs and hit
// program
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

// Source of use of Beer-Lambert Law:
// https://github.com/aromanro/RayTracer/blob/c8ad5de7fa91faa7e0a9de652c21284633659e2c/RayTracer/Material.cpp

// Beer-Lambert Law Theory:
// http://www.pci.tu-bs.de/aggericke/PC4/Kap_I/beerslaw.htm

rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  base_texture, , );
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  volume_texture, , );
rtDeclareVariable(float, ref_idx, , );
rtDeclareVariable(float, density, , );

RT_PROGRAM void closest_hit() {
  int index = hit_rec.index;
  float u = hit_rec.u, v = hit_rec.v;
  float3 P = hit_rec.p, Wo = -hit_rec.view_direction;
  float3 N = hit_rec.shading_normal;

  float3 base_color = base_texture(u, v, P, index);
  float3 volume_color = volume_texture(u, v, P, index);

  float3 outward_normal;
  float ni_over_nt;
  float cosine = dot(Wo, N);

  if (cosine > 0.f) {
    // from inside the object
    outward_normal = -N;
    ni_over_nt = ref_idx;
    cosine = ref_idx * cosine / length(Wo);

    // since it was from inside the object, compute the attenuation according
    // to the Beer-Lambert Law
    // TODO: check glass.cu from optix advanced samples
    if (density > 0.f) {
      float3 absorb = hit_rec.distance * density * volume_color;
      base_color *= expf(-absorb);
    }
  } else {
    outward_normal = N;
    ni_over_nt = 1.f / ref_idx;
    cosine = -cosine / length(Wo);
  }

  float3 refracted;
  float reflect_prob;
  if (refract(Wo, outward_normal, ni_over_nt, refracted)) {
    reflect_prob = schlick(cosine, ref_idx);
  } else
    reflect_prob = 1.f;

  // reflect or refract ray
  if (rnd(prd.seed) < reflect_prob)
    prd.direction = reflect(Wo, N);
  else
    prd.direction = refracted;

  // Assign parameters to PRD
  prd.scatterEvent = rayGotBounced;
  prd.origin = hit_rec.p;
  prd.throughput *= base_color;
  prd.isSpecular = true;
}
