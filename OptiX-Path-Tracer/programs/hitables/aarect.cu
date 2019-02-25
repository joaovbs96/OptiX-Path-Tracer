#include "../prd.cuh"

// the parameters that describe each individual rectangle
rtDeclareVariable(float, a0, , );
rtDeclareVariable(float, a1, , );
rtDeclareVariable(float, b0, , );
rtDeclareVariable(float, b1, , );
rtDeclareVariable(float, k, , );
rtDeclareVariable(int, flip, , );
rtDeclareVariable(int, index, , );

// the implicit state's ray we will intersect against
rtDeclareVariable(Ray, ray, rtCurrentRay, );

// struct used to communicate between intersection programs and hit program
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

// the per ray data we operate on
rtDeclareVariable(PerRayData, prd, rtPayload, );

// Program that performs the ray-rectangle intersection
RT_PROGRAM void hit_rect_X(int pid) {
  float t = (k - ray.origin.x) / ray.direction.x;
  if (t > ray.tmax || t < ray.tmin) return;

  float a = ray.origin.y + t * ray.direction.y;
  float b = ray.origin.z + t * ray.direction.z;
  if (a < a0 || a > a1 || b < b0 || b > b1) return;

  if (rtPotentialIntersection(t)) {
    hit_rec.distance = t;

    float3 hit_point = ray.origin + t * ray.direction;
    hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
    hit_rec.p = hit_point;

    // flip normal if needed
    float3 normal = make_float3(1.f, 0.f, 0.f);
    if (flip) normal = -normal;
    hit_rec.normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));

    hit_rec.u = (a - a0) / (a1 - a0);
    hit_rec.v = (b - b0) / (b1 - b0);

    hit_rec.index = index;

    rtReportIntersection(0);
  }
}

RT_PROGRAM void hit_rect_Y(int pid) {
  float t = (k - ray.origin.y) / ray.direction.y;
  if (t > ray.tmax || t < ray.tmin) return;

  float a = ray.origin.x + t * ray.direction.x;
  float b = ray.origin.z + t * ray.direction.z;
  if (a < a0 || a > a1 || b < b0 || b > b1) return;

  if (rtPotentialIntersection(t)) {
    hit_rec.distance = t;

    float3 hit_point = ray.origin + t * ray.direction;
    hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
    hit_rec.p = hit_point;

    // flip normal if needed
    float3 normal = make_float3(0.f, 1.f, 0.f);
    if (flip) normal = -normal;
    hit_rec.normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));

    hit_rec.u = (a - a0) / (a1 - a0);
    hit_rec.v = (b - b0) / (b1 - b0);

    hit_rec.index = index;

    rtReportIntersection(0);
  }
}

RT_PROGRAM void hit_rect_Z(int pid) {
  float t = (k - ray.origin.z) / ray.direction.z;
  if (t > ray.tmax || t < ray.tmin) return;

  float a = ray.origin.x + t * ray.direction.x;
  float b = ray.origin.y + t * ray.direction.y;
  if (a < a0 || a > a1 || b < b0 || b > b1) return;

  if (rtPotentialIntersection(t)) {
    hit_rec.distance = t;

    float3 hit_point = ray.origin + t * ray.direction;
    hit_point = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);
    hit_rec.p = hit_point;

    // flip normal if needed
    float3 normal = make_float3(0.f, 0.f, 1.f);
    if (flip) normal = -normal;
    hit_rec.normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, normal));

    hit_rec.u = (a - a0) / (a1 - a0);
    hit_rec.v = (b - b0) / (b1 - b0);

    hit_rec.index = index;

    rtReportIntersection(0);
  }
}

RT_PROGRAM void get_bounds_X(int pid, float result[6]) {
  Aabb* aabb = (Aabb*)result;

  aabb->m_min = make_float3(k - 0.0001, a0, b0);
  aabb->m_max = make_float3(k + 0.0001, a1, b1);
}

RT_PROGRAM void get_bounds_Y(int pid, float result[6]) {
  Aabb* aabb = (Aabb*)result;

  aabb->m_min = make_float3(a0, k - 0.0001, b0);
  aabb->m_max = make_float3(a1, k + 0.0001, b1);
}

RT_PROGRAM void get_bounds_Z(int pid, float result[6]) {
  Aabb* aabb = (Aabb*)result;

  aabb->m_min = make_float3(a0, b0, k - 0.0001);
  aabb->m_max = make_float3(a1, b1, k + 0.0001);
}
