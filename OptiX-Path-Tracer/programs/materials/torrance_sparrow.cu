#include "light_sample.cuh"

///////////////////////////////////////////////
// --- Torrance–Sparrow Reflaction Model --- //
///////////////////////////////////////////////

// Based on PBRT code & theory
// http://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models.html#TheTorrancendashSparrowModel
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/reflection.h#L429

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
rtDeclareVariable(float, nu, , );
rtDeclareVariable(float, nv, , );

RT_FUNCTION Torrance_Sparrow_Parameters Get_Parameters(const float3 &P, float u,
                                                       float v, int index) {
  Torrance_Sparrow_Parameters surface;

  surface.color = sample_texture(u, v, P, index);
  surface.nu = nu;
  surface.nv = nv;

  return surface;
}

// Assigns material and hit parameters to PRD
RT_PROGRAM void closest_hit() {
  HitRecord rec = Get_HitRecord(geo_index, ray, t_hit, bc);
  int index = rec.index;          // texture index
  float3 P = rec.P;               // Hit Point
  float3 Wo = rec.Wo;             // Ray view direction
  float3 N = rec.shading_normal;  // normal

  Torrance_Sparrow_Parameters surface = Get_Parameters(P, rec.u, rec.v, index);

  // Sample BRDF
  float3 Wi = Sample(surface, P, Wo, N, prd.seed);
  float pdf;  // calculated in the Evaluate function
  float3 attenuation = Evaluate(surface, P, Wo, Wi, N, pdf);

  // Assign parameters to PRD
  prd.scatterEvent = rayGotBounced;
  prd.origin = P;
  prd.direction = Wi;
  prd.throughput *= clamp(attenuation / pdf, 0.f, 1.f);
  prd.isSpecular = true;
}