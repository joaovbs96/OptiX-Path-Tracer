#include <optix_world.h>
#include "../prd.h"

/*! the parameters that describe each individual sphere geometry */
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float,  radius, , );
rtDeclareVariable(float,  density, , );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(float3, hit_rec_normal, attribute hit_rec_normal, );
rtDeclareVariable(float3, hit_rec_p, attribute hit_rec_p, );
rtDeclareVariable(float, hit_rec_u, attribute hit_rec_u, );
rtDeclareVariable(float, hit_rec_v, attribute hit_rec_v, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );

inline __device__ void get_sphere_uv(const vec3f& p) {
	float phi = atan2(p.z, p.x);
	float theta = asin(p.y); 

	hit_rec_u = 1 - (phi + CUDART_PI_F) / (2 * CUDART_PI_F);
	hit_rec_v = (theta + CUDART_PI_F / 2) / CUDART_PI_F;
}

inline __device__ bool hit_boundary(const float tmin, const float tmax, float &rec) {
  const float3 oc = ray.origin - center;

	// if the ray hits the sphere, the following equation has two roots:
	// tdot(B, B) + 2tdot(B,A-C) + dot(A-C,A-C) - R = 0

	// Using Bhaskara's Formula, we have:
  const float  a = dot(ray.direction, ray.direction);
  const float  b = dot(oc, ray.direction);
  const float  c = dot(oc, oc) - radius * radius;
  const float  discriminant = b * b - a * c;
  
  // if the descriminant is lower than zero, there's no real 
  // solution and thus no hit
  if (discriminant < 0.f) 
    return false;

  // first root of the sphere equation:
  float temp = (-b - sqrtf(discriminant)) / a;

  // for a sphere, its normal is in (hitpoint - center)
  
  // if the first root was a hit,
  if (temp < tmax && temp > tmin) {
    rec = temp;
    return true;
  }
  
  // if the second root was a hit,
  temp = (-b + sqrtf(discriminant)) / a;
  if (temp < tmax && temp > tmin){
    rec = temp;
    return true;
  }

  return false;
}

// Program that performs the ray-sphere intersection
//
// note that this is here is a simple, but not necessarily most numerically
// stable ray-sphere intersection variant out there. There are more
// stable variants out there, but for now let's stick with the one that
// the reference code used.
RT_PROGRAM void hit_sphere(int pid) {
  float rec1, rec2;

  if(hit_boundary(-FLT_MAX, FLT_MAX, rec1))
    if(hit_boundary(rec1 + 0.0001, FLT_MAX, rec2)){
      if(rec1 < ray.tmin)
        rec1 = ray.tmin;

      if(rec2 > ray.tmax)
        rec2 = ray.tmax;

      if(rec1 >= rec2)
        return;

      if(rec1 < 0.f)
        rec1 = 0.f;

      float distance_inside_boundary = rec2 - rec1;
      distance_inside_boundary *= vec3f(ray.direction).length();

      float hit_distance = -(1.f / density) * log((*prd.in.randState)());
      float temp = rec1 + hit_distance / vec3f(ray.direction).length();

      if (rtPotentialIntersection(temp)) {
        float3 hit_point = ray.origin + temp * ray.direction;
        hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
        hit_rec_p = hit_point;
  
        float3 normal = make_float3(1.f, 0.f, 0.f);
        normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
        hit_rec_normal = normal;
  
        hit_rec_u = 0.f;
        hit_rec_v = 0.f;
  
        rtReportIntersection(0);
      }
    }
}

/*! returns the bounding box of the pid'th primitive
  in this gometry. Since we only have one sphere in this 
  program (we handle multiple spheres by having a different
  geometry per sphere), the'pid' parameter is ignored */
RT_PROGRAM void get_bounds(int pid, float result[6]) {
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = center - radius;
  aabb->m_max = center + radius;
}