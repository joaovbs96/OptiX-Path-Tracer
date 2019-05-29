#include "light_sample.cuh"

//////////////////////////////////////////
// --- Oren-Nayar Reflectance Model --- //
//////////////////////////////////////////

// Original Paper: "Generalization of Lambert’s Reflectance Model"
// http://www.cs.columbia.edu/CAVE/projects/oren/
// http://www1.cs.columbia.edu/CAVE/publications/pdfs/Oren_SIGGRAPH94.pdf

// Yasuhiro Fujii’s "A tiny improvement of Oren-Nayar reflectance model" variant
// http://mimosa-pudica.net/improved-oren-nayar.html

// Reference Implementations:
// https://developer.blender.org/diffusion/C/browse/master/src/kernel/closure/bsdf_oren_nayar.h
// https://github.com/mmp/pbrt-v3/blob/f7653953b2f9cc5d6a53b46acb5ce03317fd3e8b/src/core/reflection.cpp#L197-L224

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
rtDeclareVariable(float, rA, , );
rtDeclareVariable(float, rB, , );

RT_FUNCTION Oren_Nayar_Parameters Get_Parameters(const float3 &P, float u,
                                                 float v, int index) {
  Oren_Nayar_Parameters surface;

  surface.color = sample_texture(u, v, P, index);
  surface.rA = rA;
  surface.rB = rB;

  return surface;
}

// Assigns material and hit parameters to PRD
RT_PROGRAM void closest_hit() {
  HitRecord rec = Get_HitRecord(geo_index, ray, t_hit, bc);
  int index = rec.index;          // texture index
  float3 P = rec.P;               // Hit Point
  float3 Wo = rec.Wo;             // Ray view direction
  float3 N = rec.shading_normal;  // normal

  Oren_Nayar_Parameters surface = Get_Parameters(P, rec.u, rec.v, index);

  // Sample Direct Light
  float3 direct = Direct_Light(surface, P, Wo, N, false, prd.seed);
  prd.radiance += prd.throughput * direct;

  // Sample BRDF
  float3 Wi = Sample(surface, P, Wo, N, prd.seed);
  float pdf;  // calculated in the Evaluate function
  float3 attenuation = Evaluate(surface, P, Wo, Wi, N, pdf);

  // Assign parameters to PRD
  prd.scatterEvent = rayGotBounced;
  prd.origin = P;
  prd.direction = Wi;
  prd.throughput *= clamp(attenuation / pdf, 0.f, 1.f);
  prd.isSpecular = false;
}