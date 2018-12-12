#include <optix_world.h>
#include "../prd.h"
#include "math_solvers.h"

// references:
// https://marcin-chwedczuk.github.io/ray-tracing-torus
// https://github.com/marcin-chwedczuk/ray_tracing_torus_js
// https://graphics.stanford.edu/courses/cs348b-competition/cs348b-05/donut/index.html

// Given the torus equation F(x,y,z) we want to solve it for all positive (t>0) 
// solutions to the equation:  F(r(t))=0,t>0

/*! the parameters that describe each individual sphere geometry */
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float,  sweptR, , );
rtDeclareVariable(float,  tubeR, , );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(float3, hit_rec_normal, attribute hit_rec_normal, );
rtDeclareVariable(float3, hit_rec_p, attribute hit_rec_p, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );

// Program that performs the ray-torus intersection
RT_PROGRAM void hit_torus(int pid) {
    const float3 oc = ray.origin - center;
  
    const float dSqrd = dot(ray.direction, ray.direction);
    const float 4aSqrd = 4.f * sweptR * sweptR;
    const float e = dot(ray.origin, ray.origin) - sweptR * sweptR - tubeR * tubeR;
    const float f = dot(ray.origin, ray.direction);

    const float c[5];
    c[0] = e * e - 4aSqrd * (tubeR * tubeR  - ray.origin.y * ray.origin.y);
    c[1] = 4.f * f * e + 2.f * 4aSqrd * ray.origin.y * ray.origin.y;
    c[2] = 2.f * dSqrd * e + 4.f * f * f * + 4aSqrd * ray.direction.y * ray.direction.y;
    c[3] = 4.f * dSqrd * f;
    c[4] = dSqrd * dSqrd;

    float x[4];
    int n;

    solver4(c, x, n);

    if(n <= 0)
      return;
        
    // find the smallest root greater than kEpsilon, if any
    // the roots array is not sorted
    float temp = FLT_MAX;
    for (int i = 0; i < 4; i++) {
      if (x[i] < ray.tmax && x[i] > ray.tmin) {
        temp = x[i];
      }
    }
    
    // if the second root was a hit,
    if (temp < ray.tmax && temp > ray.tmin) {
      if (rtPotentialIntersection(temp)) {
        float3 hit_point = ray.origin + temp * ray.direction;
        hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
        hit_rec_p = hit_point;
        
        float paramSqrd = sweptR * sweptR + tubeR * tubeR;
        float sumSqrd = dot(hit_rec_p, hit_rec_p);

        float3 normal = vec3f(4.f * hit_rec_p.x * (sumSqrd - paramSqrd),
                              4.f * hit_rec_p.y * (sumSqrd - paramSqrd + 2.f * sweptR * sweptR),
                              4.f * hit_rec_p.z * (sumSqrd - paramSqrd));
        normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
        hit_rec_normal = normal;

        //TODO: UV
        
        rtReportIntersection(0);
      }
    }
  }
  
  // TODO: optmization - this is the program for a sphere with radius sweptR,
  // i.e. it's still correct but not an optimal box for the torus. 
  RT_PROGRAM void get_bounds(int pid, float result[6]) {
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = center - sweptR;
    aabb->m_max = center + sweptR;
}