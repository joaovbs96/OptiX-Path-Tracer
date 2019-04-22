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
rtDeclareVariable(Ray, ray, rtCurrentRay, );                 // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );             // ray PRD
rtDeclareVariable(rtObject, world, , );                      // scene graph
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );  // from geometry

// Material Parameters
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  diffuse_color, , );
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  specular_color, , );
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
  int index = hit_rec.index;
  float u = hit_rec.u, v = hit_rec.v;
  float3 P = hit_rec.p, Wo = hit_rec.view_direction;
  float3 N = hit_rec.shading_normal;

  Ashikhmin_Shirley_Parameters surface = Get_Parameters(P, u, v, index);

  // Sample BRDF
  float3 Wi = Sample(surface, P, Wo, N, prd.seed);
  float pdf;  // calculated in the Evaluate function
  float3 attenuation = Evaluate(surface, P, Wo, Wi, N, pdf);

  // Assign parameters to PRD
  prd.scatterEvent = rayGotBounced;
  prd.origin = hit_rec.p;
  prd.direction = Wi;
  prd.throughput *= clamp(attenuation, 0.f, 1.f);
  prd.isSpecular = true;
}