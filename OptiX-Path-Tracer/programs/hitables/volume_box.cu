#include "../prd.h"

/*! the parameters that describe each individual sphere geometry */
rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );
rtDeclareVariable(float, density, , );
rtDeclareVariable(int, index, , );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(Ray, ray, rtCurrentRay, );

/*! the attributes we use to communicate between intersection programs and hit
 * program */
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );

RT_FUNCTION bool hit_boundary(const float tmin, const float tmax, float& rec) {
  float3 t0 = (boxmin - ray.origin) / ray.direction;
  float3 t1 = (boxmax - ray.origin) / ray.direction;
  float temp1 = max_component(min_vec(t0, t1));
  float temp2 = min_component(max_vec(t0, t1));

  if (temp1 > temp2) return false;

  // if the first root was a hit,
  if (temp1 < tmax && temp1 > tmin) {
    rec = temp1;
    return true;
  }

  // if the second root was a hit,
  if (temp2 < tmax && temp2 > tmin) {
    rec = temp2;
    return true;
  }

  return false;
}

RT_PROGRAM void hit_volume(int pid) {
  float rec1, rec2;

  if (hit_boundary(-FLT_MAX, FLT_MAX, rec1))
    if (hit_boundary(rec1 + 0.0001, FLT_MAX, rec2)) {
      if (rec1 < ray.tmin) rec1 = ray.tmin;

      if (rec2 > ray.tmax) rec2 = ray.tmax;

      if (rec1 >= rec2) return;

      if (rec1 < 0.f) rec1 = 0.f;

      float distance_inside_boundary = rec2 - rec1;
      distance_inside_boundary *= length(ray.direction);

      float hit_distance = -(1.f / density) * log((*prd.randState)());
      float temp = rec1 + hit_distance / length(ray.direction);

      if (rtPotentialIntersection(temp)) {
        hit_rec.distance = temp;

        float3 hit_point = ray.origin + temp * ray.direction;
        hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
        hit_rec.p = hit_point;

        float3 normal = make_float3(1.f, 0.f, 0.f);
        normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
        hit_rec.normal = normal;

        hit_rec.u = 0.f;
        hit_rec.v = 0.f;

        hit_rec.index = index;

        rtReportIntersection(0);
      }
    }
}

/*! returns the bounding box of the pid'th primitive
  in this gometry. Since we only have one sphere in this
  program (we handle multiple spheres by having a different
  geometry per sphere), the'pid' parameter is ignored */
RT_PROGRAM void get_bounds(int pid, float result[6]) {
  Aabb* aabb = (Aabb*)result;
  aabb->m_min = boxmin - make_float3(0.0001f);
  aabb->m_max = boxmax + make_float3(0.0001f);
}
