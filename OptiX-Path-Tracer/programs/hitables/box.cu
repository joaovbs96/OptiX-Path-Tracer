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

#include <optix_world.h>
#include "../prd.h"

/*! the parameters that describe each individual sphere geometry */
rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(float3, hit_rec_normal, attribute hit_rec_normal, );
rtDeclareVariable(float3, hit_rec_p, attribute hit_rec_p, );
rtDeclareVariable(float, hit_rec_u, attribute hit_rec_u, );
rtDeclareVariable(float, hit_rec_v, attribute hit_rec_v, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );

static __device__ float3 boxnormal(float t, float3 t0, float3 t1) {
  float3 neg = make_float3(t == t0.x ? 1 : 0, t == t0.y ? 1 : 0, t == t0.z ? 1 : 0);
  float3 pos = make_float3(t == t1.x ? 1 : 0, t == t1.y ? 1 : 0, t == t1.z ? 1 : 0);
  return pos - neg;
}

// if (a < b), return a, else return b
inline __device__ float ffmin(float a, float b) {
	return a < b ? a : b;
}

// if (a > b), return a, else return b
inline __device__ float ffmax(float a, float b) {
	return a > b ? a : b;
}

// return pairwise min vector
inline __device__ float3 min_vec(float3 a, float3 b) {
	return make_float3(ffmin(a.x, b.x), ffmin(a.y, b.y), ffmin(a.z, b.z));
}

// return pairwise max vector
inline __device__ float3 max_vec(float3 a, float3 b) {
	return make_float3(ffmax(a.x, b.x), ffmax(a.y, b.y), ffmax(a.z, b.z));
}

// return max component of vector
inline __device__ float max_component(float3 a){
	return ffmax(ffmax(a.x, a.y), a.z);
}

// return max component of vector
inline __device__ float min_component(float3 a){
	return ffmin(ffmin(a.x, a.y), a.z);
}


// Program that performs the ray-box intersection
RT_PROGRAM void hit_box(int pid) {
    float3 t0 = (boxmin - ray.origin) / ray.direction;
    float3 t1 = (boxmax - ray.origin) / ray.direction;
    float tmin = max_component(min_vec(t0, t1));
    float tmax = min_component(max_vec(t0, t1));

    if(tmin <= tmax) {
      bool check_second = true;
      
      if(rtPotentialIntersection(tmin)) {
        hit_rec_p = ray.origin + tmin * ray.direction;
        hit_rec_u = 0.f;
        hit_rec_v = 0.f;
        hit_rec_normal = boxnormal(tmin, t0, t1);
        
        if(rtReportIntersection(0))
            check_second = false;
      } 
      
      if(check_second) {
        if(rtPotentialIntersection(tmax)) {
            hit_rec_p = ray.origin + tmax * ray.direction;
            hit_rec_u = 0.f;
            hit_rec_v = 0.f;
            hit_rec_normal = boxnormal(tmax, t0, t1);
            rtReportIntersection(0);
        }
      }
    }
}

/*! returns the bounding box of the pid'th primitive
  in this gometry. Since we only have one sphere in this 
  program (we handle multiple spheres by having a different
  geometry per sphere), the'pid' parameter is ignored */
RT_PROGRAM void get_bounds(int pid, float result[6]) {
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = boxmin;
  aabb->m_max = boxmax;
}
