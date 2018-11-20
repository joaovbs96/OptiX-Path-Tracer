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
#include "prd.h"
#include "sampling.h"

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray,   rtCurrentRay, );
/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd,   rtPayload, );
rtDeclareVariable(rtObject,   world, , );


/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(float3, hit_rec_normal, attribute hit_rec_normal, );
rtDeclareVariable(float3, hit_rec_p, attribute hit_rec_p, );


/*! and finally - that particular material's parameters */
rtDeclareVariable(float, ref_idx, , );



/*! the actual scatter function - in Pete's reference code, that's a
  virtual function, but since we have a different function per program
  we do not need this here */
inline __device__ bool scatter(const optix::Ray &r_in,
                               vec3f &attenuation,
                               optix::Ray &scattered
                               )
{
  vec3f outward_normal;
  vec3f reflected = reflect(r_in.direction, hit_rec_normal);
  float ni_over_nt;
  attenuation = vec3f(1.f, 1.f, 1.f); 
  vec3f refracted;
  float reflect_prob;
  float cosine;
  DRand48 &rnd = *prd.randState;
  
  if (dot(r_in.direction, hit_rec_normal) > 0.f) {
    outward_normal = -hit_rec_normal;
    ni_over_nt = ref_idx;
    cosine = dot(r_in.direction, hit_rec_normal) / vec3f(r_in.direction).length();
    cosine = sqrtf(1.f - ref_idx*ref_idx*(1.f-cosine*cosine));
  }
  else {
    outward_normal = hit_rec_normal;
    ni_over_nt = 1.0 / ref_idx;
    cosine = -dot(r_in.direction, hit_rec_normal) / vec3f(r_in.direction).length();
  }
  if (refract(r_in.direction, outward_normal, ni_over_nt, refracted)) 
    reflect_prob = schlick(cosine, ref_idx);
  else 
    reflect_prob = 1.f;
  if (rnd() < reflect_prob) 
    scattered = optix::Ray(/*org */hit_rec_p,
                           /*dir */reflected.as_float3(),
                           /*type*/0,
                           /*tmin*/1e-3f,
                           /*tmax*/RT_DEFAULT_MAX);
  else 
    scattered = optix::Ray(/*org */hit_rec_p,
                           /*dir */refracted.as_float3(),
                           /*type*/0,
                           /*tmin*/1e-3f,
                           /*tmax*/RT_DEFAULT_MAX);
  return true;
}

RT_PROGRAM void closest_hit()
{
  optix::Ray scattered;
  vec3f      attenuation;
  if (prd.depth < 50 && scatter(ray,attenuation,scattered)) {
    PerRayData rec;
    rec.depth = prd.depth+1;
    rec.randState = prd.randState;
    rtTrace(world,scattered,rec);
    prd.color = attenuation * rec.color;
  } else {
    prd.color = vec3f(0,0,0);
  }
}
