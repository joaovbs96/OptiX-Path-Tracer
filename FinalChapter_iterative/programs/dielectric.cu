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
inline __device__ bool scatter(const optix::Ray &ray_in,
                               DRand48 &rnd,
                               vec3f &scattered_origin,
                               vec3f &scattered_direction,
                               vec3f &attenuation)
{
  vec3f outward_normal;
  vec3f reflected = reflect(ray_in.direction, hit_rec_normal);
  float ni_over_nt;
  attenuation = vec3f(1.f, 1.f, 1.f); 
  vec3f refracted;
  float reflect_prob;
  float cosine;
  
  if (dot(ray_in.direction, hit_rec_normal) > 0.f) {
    outward_normal = -hit_rec_normal;
    ni_over_nt = ref_idx;
    cosine = dot(ray_in.direction, hit_rec_normal) / vec3f(ray_in.direction).length();
    cosine = sqrtf(1.f - ref_idx*ref_idx*(1.f-cosine*cosine));
  }
  else {
    outward_normal = hit_rec_normal;
    ni_over_nt = 1.0 / ref_idx;
    cosine = -dot(ray_in.direction, hit_rec_normal) / vec3f(ray_in.direction).length();
  }
  if (refract(ray_in.direction, outward_normal, ni_over_nt, refracted)) 
    reflect_prob = schlick(cosine, ref_idx);
  else 
    reflect_prob = 1.f;

  scattered_origin = hit_rec_p;
  if (rnd() < reflect_prob) 
    scattered_direction = reflected;
  else 
    scattered_direction = refracted;
  
  return true;
}

RT_PROGRAM void closest_hit()
{
  prd.out.scatterEvent
    = scatter(ray,
              *prd.in.randState,
              prd.out.scattered_origin,
              prd.out.scattered_direction,
              prd.out.attenuation)
    ? rayGotBounced
    : rayGotCancelled;
}
