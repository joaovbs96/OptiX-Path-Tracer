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

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray,   rtCurrentRay, );
/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd,   rtPayload, );
rtDeclareVariable(rtObject,   world, , );


/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(float3, hit_rec_normal, attribute hit_rec_normal, );
rtDeclareVariable(float3, hit_rec_p, attribute hit_rec_p, );
rtDeclareVariable(float, hit_rec_u, attribute hit_rec_u, );
rtDeclareVariable(float, hit_rec_v, attribute hit_rec_v, );

/*! and finally - that particular material's parameters */
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, sample_texture, , );
rtDeclareVariable(float,  fuzz,   , );

/*! the actual scatter function - in Pete's reference code, that's a
  virtual function, but since we have a different function per program
  we do not need this here */
inline __device__ bool scatter(const optix::Ray &ray_in,
                               DRand48 &rndState,
                               vec3f &scattered_origin,
                               vec3f &scattered_direction,
                               vec3f &attenuation,
                               float &pdf) {
  float3 hrn = hit_rec_normal;
  vec3f reflected = reflect(unit_vector(ray_in.direction), hit_rec_normal);
  scattered_origin    = hit_rec_p;
  scattered_direction = (reflected + fuzz * random_in_unit_sphere(rndState));
  attenuation         = sample_texture(hit_rec_u, hit_rec_v, hit_rec_p);
  return (dot(scattered_direction, hrn) > 0.f);
}

inline __device__ float scattering_pdf(){
  return false;
}

inline __device__ float3 emitted(){
  return make_float3(0.f, 0.f, 0.f);
}

RT_PROGRAM void closest_hit() {
  prd.out.emitted = emitted();
  prd.out.scatterEvent
    = scatter(ray,
              *prd.in.randState,
              prd.out.scattered_origin,
              prd.out.scattered_direction,
              prd.out.attenuation,
              prd.out.pdf)
    ? rayGotBounced
    : rayGotCancelled;
  prd.out.scattered_pdf = scattering_pdf();
}
