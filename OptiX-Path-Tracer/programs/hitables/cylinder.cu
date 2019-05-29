#include "../prd.cuh"
#include "hitables.cuh"


rtDeclareVariable(float3, P0, , );  // origin
rtDeclareVariable(float3, P1, , );  // destination
rtDeclareVariable(float, R, , );
rtDeclareVariable(int, index, , );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );
rtDeclareVariable(PerRayData, prd, rtPayload, );

RT_FUNCTION bool Intersect_Plane(const float3& P, const float3& N, float& t) {
  float denominator = dot(N, ray.direction);

  if (abs(denominator) < 1e-6f) return false;

  float3 X = P - ray.origin;
  t = fabsf(dot(X, N) / denominator);

  return true;
}

RT_FUNCTION void Get_Intersection_Params(const float3& P, const float3& N,
                                         const float t) {
  hit_rec.t = t;

  hit_rec.Wo = normalize(-ray.direction);

  hit_rec.P = rtTransformPoint(RT_OBJECT_TO_WORLD, P);

  hit_rec.geometric_normal =
      normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, N));
  hit_rec.shading_normal = hit_rec.geometric_normal;

  hit_rec.index = index;

  hit_rec.u = 0;
  hit_rec.v = 0;

  rtReportIntersection(0);
}

// http://hugi.scene.org/online/hugi24/coding%20graphics%20chris%20dragan%20raytracing%20shapes.htm
RT_PROGRAM void intersection(int pid) {
  float3 X = ray.origin - P0;
  float3 V = (P1 - P0) / length(P1 - P0);  // axis

  const float DdotV = dot(ray.direction, V);
  const float DdotD = dot(ray.direction, ray.direction);
  const float DdotX = dot(ray.direction, X);
  const float XdotV = dot(X, V);
  const float XdotX = dot(X, X);

  const float a = DdotD - powf(DdotV, 2.f);
  const float b = DdotX - DdotV * XdotV;
  const float c = XdotX - powf(XdotV, 2.f) - R * R;
  const float discriminant = b * b - a * c;

  // if the discriminant is lower than zero, there's no real
  // solution and thus no hit
  if (discriminant < 0.f) return;

  // roots of the equation:
  const float t0 = (-b - sqrtf(discriminant)) / a;
  const float t1 = (-b + sqrtf(discriminant)) / a;
  float t2 = FLT_MAX, t3 = FLT_MAX;

  const bool b0 = Intersect_Plane(P0, normalize(-V), t2);
  const bool b1 = Intersect_Plane(P1, normalize(V), t3);

  // get the min T
  const float t01 = fminf(t0, t1);

  float t23 = 0.f;
  float3 planeN;
  if (t2 < t3) {
    planeN = normalize(-V);
    t23 = t2;
  } else {
    planeN = normalize(V);
    t23 = t3;
  }

  if (t01 < t23) {
    // test for the mininmal cylinder root
    if (t01 < ray.tmax && t01 > ray.tmin && rtPotentialIntersection(t01)) {
      float m = DdotV * t01 + XdotV;
      float3 hit_point = ray.origin + t01 * ray.direction;
      float3 normal = normalize(hit_point - P0 - V * m);
      Get_Intersection_Params(hit_point, normal, t01);
    }
  } else {
    // test for the minimal plane root
    if (t23 < ray.tmax && t23 > ray.tmin && rtPotentialIntersection(t23)) {
      float3 hit_point = ray.origin + t23 * ray.direction;
      Get_Intersection_Params(hit_point, planeN, t23);
    }
  }
}

// Cylinder AABB by Inigo Quilez
// http://www.iquilezles.org/www/articles/diskbbox/diskbbox.htm
RT_PROGRAM void get_bounds(int pid, float result[6]) {
  Aabb* aabb = (Aabb*)result;

  float3 a = P1 - P0;
  float3 db = R * sqrt(make_float3(1.f) - (sqr(a) / dot(a, a)));

  aabb->m_min = fminf(P0 - db, P0 - db);
  aabb->m_max = fmaxf(P1 + db, P1 + db);
}
