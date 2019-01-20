#include <optix_world.h>
#include "../prd.h"

/*! the parameters that describe each individual triangle geometry */
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float3, normal, , );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(Hit_Record, hit_rec, attribute hit_rec, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );

// Intersection program by Sam Symons - https://samsymons.com/blog/math-notes-ray-plane-intersection/
// Program that performs the ray-triangle intersection
RT_PROGRAM void hit_plane(int pid) {
  float denominator = dot(normal, ray.direction);

  if(abs(denominator) < 0.0001f)
    return;
  
  float3 diff = center - ray.origin;
  float t = dot(diff, normal) / denominator;

  if (t < ray.tmax && t > ray.tmin) {
    if (rtPotentialIntersection(t)) {
      hit_rec.distance = t;

      float3 hit_point = ray.origin + t * ray.direction;
      hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
      hit_rec.p = hit_point;

      hit_rec.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));

      hit_rec.u = hit_rec.p.x / 2500.f;
      hit_rec.v = hit_rec.p.z / 2500.f;

      hit_rec.index = 0;

      rtReportIntersection(0);
    }
  }

  return;
}

/*! returns the bounding box of the pid'th primitive in this geometry. */
RT_PROGRAM void get_bounds(int pid, float result[6]) {
  optix::Aabb* aabb = (optix::Aabb*)result;

  float x_min, y_min, z_min;
  x_min = y_min = z_min = -FLT_MAX;

  float x_max, y_max, z_max;
  x_max = y_max = z_max =  FLT_MAX;

  if(normal.x != 0) {
    x_min = center.x + 0.0001;
    x_max = center.x - 0.0001;
  }
  if(normal.y != 0) {
    y_min = center.y + 0.0001;
    y_max = center.y - 0.0001;
  }
  if(normal.z != 0) {
    z_min = center.z + 0.0001;
    z_max = center.z - 0.0001;
  }

  aabb->m_min = make_float3(x_min, y_min, z_min);
  aabb->m_max = make_float3(x_max, y_max, z_max);
}
