#include "../prd.cuh"
#include "hitables.cuh"

//////////////////
// --- Cone --- //
//////////////////


// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );

// Intersected Geometry Attributes
rtDeclareVariable(int, geo_index, attribute geo_index, );  // primitive index
rtDeclareVariable(float2, bc, attribute bc, );  // triangle barycentrics

// Primitive Parameters

rtDeclareVariable(float3, P0, , );  // origin
rtDeclareVariable(float3, P1, , );  // destination
rtDeclareVariable(float, r0, , );
rtDeclareVariable(float, r1, , );
rtDeclareVariable(int, index, , );

RT_PROGRAM void hit_sphere(int pid) {
  float3 V = (P1 - P0) / length(P1 - P0);  // axis

  float3 X = ray.origin - P0;

  float DdotV = dot(ray.direction, V);
  float DdotD = dot(ray.direction, ray.direction);
  float DdotX = dot(ray.direction, X);
  float XdotV = dot(X, V);
  float XdotX = dot(X, X);

  float a = DdotD - powf(DdotV, 2.f);
  float b = DdotX - DdotV * XdotV;
  float c = XdotX - powf(XdotV, 2.f) - r * r;
}


RT_PROGRAM void get_bounds(int pid, float result[6]) {
  Aabb* aabb = (Aabb*)result;
  aabb->m_min = center - radius;
  aabb->m_max = center + radius;
}
