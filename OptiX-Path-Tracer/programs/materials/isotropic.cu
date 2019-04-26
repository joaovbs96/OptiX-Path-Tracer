#include "light_sample.cuh"

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

rtDeclareVariable(Texture_Function, sample_texture, , );

RT_FUNCTION Isotropic_Parameters Get_Parameters(const float3 &P, float u,
                                                float v, int index) {
  Isotropic_Parameters surface;

  surface.color = sample_texture(u, v, P, index);

  return surface;
}

// Isotropic Material Closest Hit Program
RT_PROGRAM void closest_hit() {
  int index = hit_rec.index;
  float u = hit_rec.u, v = hit_rec.v;
  float3 P = hit_rec.p, Wo = hit_rec.view_direction;
  float3 N = hit_rec.shading_normal;

  Isotropic_Parameters surface = Get_Parameters(P, u, v, index);

  // Sample Direct Light
  float3 direct = Direct_Light(surface, P, Wo, N, false, prd.seed);
  prd.radiance += prd.throughput * direct;

  // Sample BRDF
  float3 Wi = Sample(surface, P, Wo, N, prd.seed);
  float pdf;  // calculated in the Evaluate function
  float3 attenuation = Evaluate(surface, P, Wo, Wi, N, pdf);

  // Assign parameters to PRD
  prd.scatterEvent = rayGotBounced;
  prd.origin = hit_rec.p;
  prd.direction = Wi;
  prd.throughput *= attenuation / pdf;
  prd.isSpecular = false;
}
