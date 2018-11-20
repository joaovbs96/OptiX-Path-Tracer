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
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );


/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(float3, hit_rec_normal, attribute hit_rec_normal, );
rtDeclareVariable(float3, hit_rec_p, attribute hit_rec_p, );


/*! and finally - that particular material's parameters */
rtDeclareVariable(float3, albedo, , );



/*! the actual scatter function - in Pete's reference code, that's a
  virtual function, but since we have a different function per program
  we do not need this here */
inline __device__ bool scatter(const optix::Ray &ray_in,
                               vec3f &attenuation,
                               optix::Ray &scattered)
{
  vec3f target = hit_rec_p + hit_rec_normal + random_in_unit_sphere(*prd.randState);
  scattered    = optix::Ray(/*org */hit_rec_p,
                            /*dir */(target-hit_rec_p).as_float3(),
                            /*type*/0,
                            /*tmin*/1e-3f,
                            /*tmax*/RT_DEFAULT_MAX);
  attenuation  = albedo;
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
