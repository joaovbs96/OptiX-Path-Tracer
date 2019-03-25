#include "hitables.cuh"

/*! the parameters that describe each individual triangle geometry */
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float3, normal, , );
rtDeclareVariable(int, index, , );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(Ray, ray, rtCurrentRay, );

/*! the attributes we use to communicate between intersection programs and hit
 * program */
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );

// Intersection program by Sam Symons -
// https://samsymons.com/blog/math-notes-ray-plane-intersection/
// Program that performs the ray-plane intersection
RT_PROGRAM void hit_plane(int pid) {
  float denominator = dot(normal, ray.direction);

  if (abs(denominator) < 0.0001f) return;

  float3 diff = center - ray.origin;
  float t = dot(diff, normal) / denominator;

  if (t < ray.tmax && t > ray.tmin) {
    if (rtPotentialIntersection(t)) {
      hit_rec.distance = t;

      hit_rec.view_direction = normalize(-ray.direction);

      float3 hit_point = ray.origin + t * ray.direction;
      hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
      hit_rec.p = hit_point;

      float3 pN = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
      hit_rec.geometric_normal = pN;
      hit_rec.shading_normal = hit_rec.geometric_normal;

      // TODO: fix
      if (normal.x != 0) {
        hit_rec.u = hit_rec.p.y / 2500.f;
        hit_rec.v = hit_rec.p.z / 2500.f;
      } else if (normal.y != 0) {
        hit_rec.u = hit_rec.p.x / 2500.f;
        hit_rec.v = hit_rec.p.z / 2500.f;
      } else {
        hit_rec.u = hit_rec.p.x / 2500.f;
        hit_rec.v = hit_rec.p.y / 2500.f;
      }

      hit_rec.index = index;

      rtReportIntersection(0);
    }
  }

  return;
}

/*! returns the bounding box of the pid'th primitive in this geometry. */
RT_PROGRAM void get_bounds(int pid, float result[6]) {
  Aabb* aabb = (Aabb*)result;

  float x_min, y_min, z_min;
  x_min = y_min = z_min = -FLT_MAX;

  float x_max, y_max, z_max;
  x_max = y_max = z_max = FLT_MAX;

  if (normal.x != 0) {
    x_min = center.x + 0.0001;
    x_max = center.x - 0.0001;
  }
  if (normal.y != 0) {
    y_min = center.y + 0.0001;
    y_max = center.y - 0.0001;
  }
  if (normal.z != 0) {
    z_min = center.z + 0.0001;
    z_max = center.z - 0.0001;
  }

  aabb->m_min = make_float3(x_min, y_min, z_min);
  aabb->m_max = make_float3(x_max, y_max, z_max);
}
