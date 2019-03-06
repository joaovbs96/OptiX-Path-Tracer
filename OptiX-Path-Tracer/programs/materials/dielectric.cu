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
  prd.matType = Dielectric_Material;
  prd.isSpecular = true;
  prd.scatterEvent = rayGotBounced;

  prd.origin = hit_rec.p;
  prd.normal = hit_rec.normal;

  int index = hit_rec.index;
  prd.emitted = make_float3(0.f);
  prd.attenuation = base_texture(hit_rec.u, hit_rec.v, hit_rec.p, index);
  float3 volumeColor = volume_texture(hit_rec.u, hit_rec.v, hit_rec.p, index);

  float3 outward_normal;
  float ni_over_nt;
  float cosine = dot(ray.direction, prd.normal);

  if (cosine > 0.f) {
    // from inside the object
    outward_normal = -1 * prd.normal;
    ni_over_nt = ref_idx;
    cosine = ref_idx * cosine / length(ray.direction);

    // since it was from inside the object, compute the attenuation according
    // to the Beer-Lambert Law
    // TODO: check glass.cu from optix advanced samples
    if (density > 0.f) {
      float3 absorb = hit_rec.distance * density * volumeColor;
      prd.attenuation *= expf(-absorb);
    }
  }

  else {
    outward_normal = prd.normal;
    ni_over_nt = 1.f / ref_idx;
    cosine = -cosine / length(ray.direction);
  }

  float3 refracted;
  float reflect_prob;
  if (refract(ray.direction, outward_normal, ni_over_nt, refracted)) {
    reflect_prob = schlick(cosine, ref_idx);
  } else
    reflect_prob = 1.f;

  float3 reflected = reflect(ray.direction, prd.normal);
  if (rnd(prd.seed) < reflect_prob)
    prd.direction = reflected;
  else
    prd.direction = refracted;
}

RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams& pdf, uint& seed) {
  return make_float3(1.f);
}

RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams& pdf) { return 1.f; }

RT_CALLABLE_PROGRAM float BRDF_Evaluate(PDFParams& pdf) { return 1.f; }
