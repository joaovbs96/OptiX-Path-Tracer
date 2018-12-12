#include <optix_world.h>
#include "../prd.h"

// references:
// AABB intersection function from Peter Shirley's "The Next Week"
// Box intersection function from the optixTutorial sample from OptiX's SDK

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

// Program that performs the ray-box intersection
RT_PROGRAM void hit_box(int pid) {
    float3 t0 = (boxmin - ray.origin) / ray.direction;
    float3 t1 = (boxmax - ray.origin) / ray.direction;
    float tmin = max_component(min_vec(t0, t1));
    float tmax = min_component(max_vec(t0, t1));

    if(tmin <= tmax) {
      bool check_second = true;
      
      if(rtPotentialIntersection(tmin)) {
        float3 hit_point = ray.origin + tmin * ray.direction;
        hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
        hit_rec_p = hit_point;
        
        hit_rec_u = 0.f;
        hit_rec_v = 0.f;

        float3 normal = boxnormal(tmin, t0, t1);
        normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
        hit_rec_normal = normal;
        
        if(rtReportIntersection(0))
            check_second = false;
      } 
      
      if(check_second) {
        if(rtPotentialIntersection(tmax)) {
            float3 hit_point = ray.origin + tmax * ray.direction;
            hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
            hit_rec_p = hit_point;

            hit_rec_u = 0.f;
            hit_rec_v = 0.f;
            
            float3 normal = boxnormal(tmax, t0, t1);
            normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
            hit_rec_normal = normal;
            
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
