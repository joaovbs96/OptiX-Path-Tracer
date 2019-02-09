#include "../prd.h"

// references:
// AABB intersection function from Peter Shirley's "The Next Week"
// Box intersection function from the optixTutorial sample from OptiX's SDK

/*! the parameters that describe each individual sphere geometry */
rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(Ray, ray, rtCurrentRay, );

/*! the attributes we use to communicate between intersection programs and hit
 * program */
rtDeclareVariable(Hit_Record, hit_rec, attribute hit_rec, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );

RT_FUNCTION float3 boxnormal(float t, float3 t0, float3 t1) {
  float3 neg =
      make_float3(t == t0.x ? 1 : 0, t == t0.y ? 1 : 0, t == t0.z ? 1 : 0);
  float3 pos =
      make_float3(t == t1.x ? 1 : 0, t == t1.y ? 1 : 0, t == t1.z ? 1 : 0);
  return pos - neg;
}

// Program that performs the ray-box intersection
RT_PROGRAM void hit_box(int pid) {
  float3 t0 = (boxmin - ray.origin) / ray.direction;
  float3 t1 = (boxmax - ray.origin) / ray.direction;
  float tmin = max_component(min_vec(t0, t1));
  float tmax = min_component(max_vec(t0, t1));

  if (tmin <= tmax) {
    bool check_second = true;

    if (rtPotentialIntersection(tmin)) {
      hit_rec.distance = tmin;

      float3 hit_point = ray.origin + tmin * ray.direction;
      hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
      hit_rec.p = hit_point;

      hit_rec.u = 0.f;
      hit_rec.v = 0.f;

      float3 normal = boxnormal(tmin, t0, t1);
      normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
      hit_rec.normal = normal;

      hit_rec.index = 0;

      if (rtReportIntersection(0)) check_second = false;
    }

    if (check_second) {
      if (rtPotentialIntersection(tmax)) {
        hit_rec.distance = tmax;

        float3 hit_point = ray.origin + tmax * ray.direction;
        hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
        hit_rec.p = hit_point;

        hit_rec.u = 0.f;
        hit_rec.v = 0.f;

        float3 normal = boxnormal(tmax, t0, t1);
        normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
        hit_rec.normal = normal;

        hit_rec.index = 0;

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
  Aabb* aabb = (Aabb*)result;
  aabb->m_min = boxmin;
  aabb->m_max = boxmax;
}
