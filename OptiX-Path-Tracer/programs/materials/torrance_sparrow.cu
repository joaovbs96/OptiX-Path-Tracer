#include "material.cuh"
#include "microfacets.cuh"

///////////////////////////////////////////////
// --- Torrance–Sparrow Reflaction Model --- //
///////////////////////////////////////////////

// Based on PBRT code & theory
// http://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models.html#TheTorrancendashSparrowModel
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/reflection.h#L429

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                 // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );             // ray PRD
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );  // from geometry

// Material Parameters
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  sample_texture, , );
rtDeclareVariable(float, nu, , );
rtDeclareVariable(float, nv, , );

///////////////////////////
// --- BRDF Programs --- //
///////////////////////////

// Assigns material and hit parameters to PRD
RT_PROGRAM void closest_hit() {
  prd.matType = Torrance_Sparrow_BRDF;
  prd.isSpecular = false;
  prd.scatterEvent = rayGotBounced;

  // Get hit params
  prd.origin = hit_rec.p;
  prd.geometric_normal = normalize(hit_rec.geometric_normal);
  prd.shading_normal = normalize(hit_rec.shading_normal);
  prd.view_direction = normalize(hit_rec.view_direction);

  // Get material color
  int index = hit_rec.index;
  float3 color = sample_texture(hit_rec.u, hit_rec.v, hit_rec.p, index);

  // Assign material parameters to PRD, to be used in the sampling programs
  prd.matParams.attenuation = color;
  prd.matParams.anisotropic.nu = nu;
  prd.matParams.anisotropic.nv = nv;
}

// Samples BRDF, generating outgoing direction(Wo)
RT_CALLABLE_PROGRAM float3 BRDF_Sample(const BRDFParameters &surface,
                                       const float3 &P,   // next ray origin
                                       const float3 &Wo,  // prev ray direction
                                       const float3 &N,   // shading normal
                                       uint &seed) {
  // Get material params from input variable
  float nu = surface.anisotropic.nu;
  float nv = surface.anisotropic.nv;

  // create basis
  float3 Nn = normalize(N);
  float3 T = normalize(cross(Nn, make_float3(0.f, 1.f, 0.f)));
  float3 B = cross(T, Nn);

  // random variables
  float2 random = make_float2(rnd(seed), rnd(seed));

  // get half vector and rotate it to world space
  float3 H = normalize(GGX_Sample(Wo, random, nu, nv));
  H = H.x * B + H.y * Nn + H.z * T;

  float HdotI = dot(H, Wo);
  if (HdotI < 0.f) H = -H;

  return normalize(-Wo + 2.f * dot(Wo, H) * H);
}

// Gets BRDF PDF value
RT_CALLABLE_PROGRAM float BRDF_PDF(const BRDFParameters &surface,
                                   const float3 &P,    // next ray origin
                                   const float3 &Wo,   // prev ray direction
                                   const float3 &Wi,   // next ray direction
                                   const float3 &N) {  // shading normal
  // Get material params from input variable
  float nu = surface.anisotropic.nu;
  float nv = surface.anisotropic.nv;

  // Handles degenerate cases for microfacet reflection
  float3 H = normalize(Wi + Wo);

  return GGX_PDF(H, Wo, nu, nv) / (4.f * dot(Wo, H));
}

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3
BRDF_Evaluate(const BRDFParameters &surface,
              const float3 &P,    // next ray origin
              const float3 &Wo,   // prev ray direction
              const float3 &Wi,   // next ray direction
              const float3 &N) {  // shading normal
  // Get material params from input variable
  float3 Rs = surface.attenuation;
  float nu = surface.anisotropic.nu;
  float nv = surface.anisotropic.nv;

  // create basis
  float3 Up = make_float3(0.f, 1.f, 0.f);
  float NdotI = fmaxf(dot(Up, Wi), 1e-6f), NdotO = fmaxf(dot(Up, Wo), 1e-6f);

  // half vector = (v1 + v2) / |v1 + v2|
  float3 H = Wo + Wi;
  if (isNull(H)) return make_float3(0.f);
  H = normalize(H);
  float HdotI = abs(dot(H, Wi));  // origin or direction here

  float3 F = schlick(Rs, HdotI);    // Fresnel Reflectance
  float G = GGX_G(Wo, Wi, nu, nv);  // Geometric Shadowing
  float D = GGX_D(H, nu, nv);       // Normal Distribution Function(NDF)

  return Rs * D * G * F / (4.f * NdotI * NdotO);
}