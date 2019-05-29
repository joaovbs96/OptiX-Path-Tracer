#include "../prd.cuh"
#include "hitables.cuh"

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );

// Intersected Geometry Attributes
rtDeclareVariable(int, geo_index, attribute geo_index, );  // primitive index
rtDeclareVariable(float2, bc, attribute bc, );  // triangle barycentrics

// Primitive Parameters
rtDeclareVariable(int, axis, , );
rtDeclareVariable(float, a0, , );
rtDeclareVariable(float, a1, , );
rtDeclareVariable(float, b0, , );
rtDeclareVariable(float, b1, , );
rtDeclareVariable(float, k, , );
rtDeclareVariable(int, flip, , );

RT_FUNCTION bool Hit_X(float& t) {
  t = (k - ray.origin.x) / ray.direction.x;
  float a = ray.origin.y + t * ray.direction.y;
  float b = ray.origin.z + t * ray.direction.z;
  if (a < a0 || a > a1 || b < b0 || b > b1) return false;

  return true;
}

RT_FUNCTION bool Hit_Y(float& t) {
  t = (k - ray.origin.y) / ray.direction.y;
  float a = ray.origin.x + t * ray.direction.x;
  float b = ray.origin.z + t * ray.direction.z;
  if (a < a0 || a > a1 || b < b0 || b > b1) return false;

  return true;
}

RT_FUNCTION bool Hit_Z(float& t) {
  t = (k - ray.origin.z) / ray.direction.z;
  float a = ray.origin.x + t * ray.direction.x;
  float b = ray.origin.y + t * ray.direction.y;
  if (a < a0 || a > a1 || b < b0 || b > b1) return false;

  return true;
}

RT_PROGRAM void Hit_Rect(int pid) {
  bool hit = false;
  float t;
  switch (AXIS(axis)) {
    case X_AXIS:
      hit = Hit_X(t);
      break;
    case Y_AXIS:
      hit = Hit_Y(t);
      break;
    case Z_AXIS:
      hit = Hit_Z(t);
      break;
    default:
      printf("Error: invalid axis");
  }

  if (hit && rtPotentialIntersection(t)) {
    geo_index = 0;
    bc = make_float2(0);
    rtReportIntersection(0);
  }
}

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

  // Get normal and texture coordinates depending on axis
  float3 normal;
  switch (AXIS(axis)) {
    case X_AXIS:
      normal = make_float3(1.f, 0.f, 0.f);
      rec.u = (hit_point.y - a0) / (a1 - a0);
      rec.v = (hit_point.z - b0) / (b1 - b0);
      break;
    case Y_AXIS:
      normal = make_float3(0.f, 1.f, 0.f);
      rec.u = (hit_point.x - a0) / (a1 - a0);
      rec.v = (hit_point.z - b0) / (b1 - b0);
      break;
    case Z_AXIS:
      normal = make_float3(0.f, 0.f, 1.f);
      rec.u = (hit_point.x - a0) / (a1 - a0);
      rec.v = (hit_point.y - b0) / (b1 - b0);
      break;
    default:
      printf("Error: invalid axis");
  }

  // Normal
  normal = flip ? -normal : normal;
  normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));
  rec.shading_normal = rec.geometric_normal = normal;

  // Texture Index
  rec.index = index;

  return rec;
}

RT_PROGRAM void Get_Bounds(int pid, float result[6]) {
  Aabb* aabb = (Aabb*)result;

  switch (AXIS(axis)) {
    case X_AXIS:
      aabb->m_min = make_float3(k - 0.0001f, a0, b0);
      aabb->m_max = make_float3(k + 0.0001f, a1, b1);
      break;
    case Y_AXIS:
      aabb->m_min = make_float3(a0, k - 0.0001f, b0);
      aabb->m_max = make_float3(a1, k + 0.0001f, b1);
      break;
    case Z_AXIS:
      aabb->m_min = make_float3(a0, b0, k - 0.0001f);
      aabb->m_max = make_float3(a1, b1, k + 0.0001f);
      break;
    default:
      printf("Error: invalid axis");
  }
}