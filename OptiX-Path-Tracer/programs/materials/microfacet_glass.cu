#include "light_sample.cuh"

////////////////////////////////////
// --- Microfacet Glass Model --- //
////////////////////////////////////

// Based on PBRT code & theory
// http://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models.html#TheTorrancendashSparrowModel
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/reflection.h#L429

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                 // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );             // ray PRD
rtDeclareVariable(rtObject, world, , );                      // scene graph
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );  // from geometry

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
  int index = hit_rec.index;
  float u = hit_rec.u, v = hit_rec.v;
  float3 P = hit_rec.p, Wo = hit_rec.view_direction;
  float3 N = hit_rec.shading_normal;

  Torrance_Sparrow_Parameters surface = Get_Parameters(P, u, v, index);

  // Sample BRDF
  float3 Wi = Sample(surface, P, Wo, N, prd.seed);
  float pdf;  // calculated in the Evaluate function
  float3 attenuation = Evaluate(surface, P, Wo, Wi, N, pdf);

  // Assign parameters to PRD
  prd.scatterEvent = rayGotBounced;
  prd.origin = hit_rec.p;
  prd.direction = Wi;
  prd.throughput *= clamp(attenuation / pdf, 0.f, 1.f);
  prd.isSpecular = true;
}

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3 BRDF_Evaluate(PDFParams &pdf) {
  float3 Wo = pdf.view_direction, Wi = pdf.direction;
  float3 Rs = pdf.matParams.attenuation;
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;
  
  float cosThetaI = CosTheta(Wi), cosThetaO = CosTheta(Wo);
  if (cosThetaI == 0.f || cosThetaO == 0.f) return make_float3(0.f);

  float eta = CosTheta(Wo) > 0 ? (nv / nu) : (nu / nv);
  float3 H = normalize(Wo + Wi * eta);
  if(H.y < 0) H *= -1;

  // *Note that this is not a symetric BRDF. The PBRT implementation take both 
  // directions into account. The code here works only on unidirectional PTs.
  // https://github.com/mmp/pbrt-v3/blob/f7653953b2f9cc5d6a53b46acb5ce03317fd3e8b/src/core/reflection.cpp#L260

  float HdotO = dot(H, Wo), HdotI = dot(H, Wi);
  float AHdotO = fabsf(HdotO), AHdotI = fabsf(HdotI);

  float denom = Square(HdotO + eta * HdotI) * cosThetaI * cosThetaO;

  float3 F = schlick(Rs, HdotI);    // Fresnel Reflectance
  float G = GGX_G(Wo, Wi, nu, nv);  // Geometric Shadowing
  float D = GGX_D(H, nu, nv);       // Normal Distribution Function(NDF)

  return (Rs * (make_float3(1.f) - F) * D * G  * AHdotI * AHdotO) / denom;
}