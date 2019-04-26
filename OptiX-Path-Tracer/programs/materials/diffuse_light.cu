#include "light_sample.cuh"

//////////////////////////////////////////
// --- Diffuse Light Material Model --- //
//////////////////////////////////////////

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                 // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );             // ray PRD
rtDeclareVariable(rtObject, world, , );                      // scene graph
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );  // from geometry

// Material Parameters
rtDeclareVariable(Texture_Function, sample_texture, , );

RT_FUNCTION Diffuse_Light_Parameters Get_Parameters(const float3 &P, float u,
                                                    float v, int index) {
  Diffuse_Light_Parameters surface;

  surface.color = sample_texture(u, v, P, index);

  return surface;
}

RT_PROGRAM void closest_hit() {
  int index = hit_rec.index;
  float u = hit_rec.u, v = hit_rec.v;
  float3 P = hit_rec.p, Wo = hit_rec.view_direction;
  float3 N = hit_rec.shading_normal;

  Diffuse_Light_Parameters surface = Get_Parameters(P, u, v, index);

  // Sample Direct Light
  float3 direct = Direct_Light(surface, P, Wo, N, true, prd.seed);
  prd.radiance += prd.throughput * direct;

  // Take Light emission into account
  if (dot(N, Wo) < 0.f) prd.throughput *= surface.color;

  // Assign parameters to PRD
  prd.scatterEvent = rayHitLight;
}