#include <optix_world.h>
#include "../prd.h"

/*! the parameters that describe each individual rectangle */
rtDeclareVariable(float,  a0, , );
rtDeclareVariable(float,  a1, , );
rtDeclareVariable(float,  b0, , );
rtDeclareVariable(float,  b1, , );
rtDeclareVariable(float,  k, , );
rtDeclareVariable(int, flip_normal, , );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(float3, hit_rec_normal, attribute hit_rec_normal, );
rtDeclareVariable(float3, hit_rec_p, attribute hit_rec_p, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );

// Program that performs the ray-sphere intersection
//
// note that this is here is a simple, but not necessarily most numerically
// stable ray-sphere intersection variant out there. There are more
// stable variants out there, but for now let's stick with the one that
// the reference code used.
RT_PROGRAM void hit_rect_X(int pid) {
    float t = (k - ray.origin.x) / ray.direction.x;
    float a = ray.origin.y + t * ray.direction.y;
    float b = ray.origin.z + t * ray.direction.z;
    float3 normal = make_float3(flip_normal * 1.f, 0.f, 0.f);

	if (a < a0 || a > a1 || b < b0 || b > b1)
        return;
        
    if (t < ray.tmax && t > ray.tmin) {
        if (rtPotentialIntersection(t)) {
          hit_rec_p = ray.origin + t * ray.direction;
          hit_rec_normal = normal;
          rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void hit_rect_Y(int pid) {
    float t = (k - ray.origin.y) / ray.direction.y;
    float a = ray.origin.x + t * ray.direction.x;
    float b = ray.origin.z + t * ray.direction.z;
    float3 normal = make_float3(0.f, flip_normal * 1.f, 0.f);

	if (a < a0 || a > a1 || b < b0 || b > b1)
        return;
    
    // rtPotentialIntersection will determine whether the 
    // reported hit distance is within the valid interval 
    // associated with the ray, and return true if the 
    // intersection is valid.
    if (t < ray.tmax && t > ray.tmin) {
        if (rtPotentialIntersection(t)) {
          hit_rec_p = ray.origin + t * ray.direction;
          hit_rec_normal = normal;
          rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void hit_rect_Z(int pid) {
    float t = (k - ray.origin.z) / ray.direction.z;
    float a = ray.origin.x + t * ray.direction.x;
    float b = ray.origin.y + t * ray.direction.y;
    float3 normal = make_float3(0.f, 0.f, flip_normal * 1.f);

	if (a < a0 || a > a1 || b < b0 || b > b1)
        return;
        
    if (t < ray.tmax && t > ray.tmin) {
        if (rtPotentialIntersection(t)) {
          hit_rec_p = ray.origin + t * ray.direction;
          hit_rec_normal = normal;
          rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void get_bounds_X(int pid, float result[6]) {
  optix::Aabb* aabb = (optix::Aabb*)result;

  aabb->m_min = make_float3(k - 0.0001, a0, b0);
  aabb->m_max = make_float3(k + 0.0001, a1, b1);
}

RT_PROGRAM void get_bounds_Y(int pid, float result[6]) {
    optix::Aabb* aabb = (optix::Aabb*)result;
  
    aabb->m_min = make_float3(a0, k - 0.0001, b0);
    aabb->m_max = make_float3(a1, k + 0.0001, b1);
}

RT_PROGRAM void get_bounds_Z(int pid, float result[6]) {
    optix::Aabb* aabb = (optix::Aabb*)result;
  
    aabb->m_min = make_float3(a0, b0, k - 0.0001);
    aabb->m_max = make_float3(a1, b1, k + 0.0001);
}

