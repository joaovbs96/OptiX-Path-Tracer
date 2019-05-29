#include "../prd.cuh"
#include "hitables.cuh"

//////////////////////////////
// --- Axis Aligned Box --- //
//////////////////////////////

// Based AABB intersection function from Peter Shirley's "The Next Week" and the
// Box intersection function from the optixTutorial sample from OptiX's SDK.

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );

// Intersected Geometry Attributes
rtDeclareVariable(int, geo_index, attribute geo_index, );  // primitive index
rtDeclareVariable(float2, bc, attribute bc, );  // triangle barycentrics

// Primitive Parameters
rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );

RT_FUNCTION float3 boxnormal(float t, Ray ray) {
  float3 t0 = (boxmin - ray.origin) / ray.direction;
  float3 t1 = (boxmax - ray.origin) / ray.direction;

  float3 neg =
      make_float3(t == t0.x ? 1 : 0, t == t0.y ? 1 : 0, t == t0.z ? 1 : 0);
  float3 pos =
      make_float3(t == t1.x ? 1 : 0, t == t1.y ? 1 : 0, t == t1.z ? 1 : 0);
  return pos - neg;
}

// Program that performs the ray-box intersection
RT_PROGRAM void Intersect(int pid) {
  float3 t0 = (boxmin - ray.origin) / ray.direction;
  float3 t1 = (boxmax - ray.origin) / ray.direction;
  float tmin = max_component(min_vec(t0, t1));
  float tmax = min_component(max_vec(t0, t1));

  if (tmin <= tmax && rtPotentialIntersection(tmin)) {
    geo_index = 0;
    bc = make_float2(0);
    rtReportIntersection(0);
  } else if (tmin <= tmax && rtPotentialIntersection(tmax)) {
    geo_index = 0;
    bc = make_float2(0);
    rtReportIntersection(0);
  }
}

// Gets HitRecord parameters, given a ray, an index and a hit distance
RT_CALLABLE_PROGRAM HitRecord Get_HitRecord(int index,    // primitive index
                                            Ray ray,      // current ray
                                            float t_hit,  // intersection dist
                                            float2 bc) {  // barycentrics
  HitRecord rec;

  // view direction
  rec.Wo = normalize(-ray.direction);

  // Hit Point
  float3 hit_point = ray.origin + t_hit * ray.direction;
  rec.P = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);

  // Normal
  float3 normal = boxnormal(t_hit, ray);
  normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
  rec.shading_normal = rec.geometric_normal = normal;

  // Texture coordinates
  rec.u = rec.v = 0.f;

  // Texture Index
  rec.index = index;

  return rec;
}

// returns the bounding box of the pid'th primitive in this gometry.
RT_PROGRAM void Get_Bounds(int pid, float result[6]) {
  Aabb* aabb = (Aabb*)result;
  aabb->m_min = boxmin;
  aabb->m_max = boxmax;
}
