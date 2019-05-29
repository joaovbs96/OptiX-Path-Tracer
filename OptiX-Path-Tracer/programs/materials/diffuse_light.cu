#include "light_sample.cuh"

//////////////////////////////////////////
// --- Diffuse Light Material Model --- //
//////////////////////////////////////////

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );            // ray PRD
rtDeclareVariable(rtObject, world, , );                     // scene graph
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );  // hit distance

// Intersected Geometry Parameters
rtDeclareVariable(HitRecord_Function, Get_HitRecord, , );  // HitRecord function
rtDeclareVariable(int, geo_index, attribute geo_index, );  // primitive index
rtDeclareVariable(float2, bc, attribute bc, );  // triangle barycentrics

// Material Parameters
rtDeclareVariable(Texture_Function, sample_texture, , );

RT_FUNCTION Diffuse_Light_Parameters Get_Parameters(const float3 &P, float u,
                                                    float v, int index) {
  Diffuse_Light_Parameters surface;

  surface.color = sample_texture(u, v, P, index);

  return surface;
}

RT_PROGRAM void closest_hit() {
  HitRecord rec = Get_HitRecord(geo_index, ray, t_hit, bc);
  int index = rec.index;          // texture index
  float3 P = rec.P;               // Hit Point
  float3 Wo = rec.Wo;             // Ray view direction
  float3 N = rec.shading_normal;  // normal

  Diffuse_Light_Parameters surface = Get_Parameters(P, rec.u, rec.v, index);

  // Sample Direct Light
  float3 direct = Direct_Light(surface, P, Wo, N, true, prd.seed);
  prd.radiance += prd.throughput * direct;

  // Take Light emission into account
  if (dot(N, Wo) < 0.f) prd.throughput *= surface.color;

  // Assign parameters to PRD
  prd.scatterEvent = rayHitLight;
}