#include "light_sample.cuh"

////////////////////////////////////////////////////////////
// --- Ashikhmin-Shirley Anisotropic Phong BRDF Model --- //
////////////////////////////////////////////////////////////

// Original Paper & Tech Report - "An Anisotropic Phong Light Reflection Model"
// https://www.cs.utah.edu/~shirley/papers/jgtbrdf.pdf
// https://www.cs.utah.edu/docs/techreports/2000/pdf/UUCS-00-014.pdf

// Reference Implementation:
// https://developer.blender.org/diffusion/C/browse/master/src/kernel/closure/bsdf_ashikhmin_shirley.h
// FresnelBlend from PBRT
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/reflection.cpp
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/reflection.h

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
rtDeclareVariable(Texture_Function, diffuse_color, , );
rtDeclareVariable(Texture_Function, specular_color, , );
rtDeclareVariable(float, nu, , );
rtDeclareVariable(float, nv, , );

RT_FUNCTION Ashikhmin_Shirley_Parameters Get_Parameters(const float3 &P,
                                                        float u, float v,
                                                        int index) {
  Ashikhmin_Shirley_Parameters surface;

  surface.diffuse_color = diffuse_color(u, v, P, index);
  surface.specular_color = specular_color(u, v, P, index);
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

  Ashikhmin_Shirley_Parameters surface = Get_Parameters(P, rec.u, rec.v, index);

  // Sample BRDF
  float3 Wi = Sample(surface, P, Wo, N, prd.seed);
  float pdf;  // calculated in the Evaluate function
  float3 attenuation = Evaluate(surface, P, Wo, Wi, N, pdf);

  // Assign parameters to PRD
  prd.scatterEvent = rayGotBounced;
  prd.origin = P;
  prd.direction = Wi;
  prd.throughput *= clamp(attenuation, 0.f, 1.f);
  prd.isSpecular = true;
}