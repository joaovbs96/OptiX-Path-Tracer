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
  prd.geometric_normal = hit_rec.geometric_normal;
  prd.shading_normal = hit_rec.shading_normal;
  prd.view_direction = hit_rec.view_direction;

  // Get material color
  int index = hit_rec.index;
  float3 color = sample_texture(hit_rec.u, hit_rec.v, hit_rec.p, index);

  // Assign material parameters to PRD, to be used in the sampling programs
  prd.matParams.attenuation = color;
  prd.matParams.anisotropic.nu = nu;
  prd.matParams.anisotropic.nv = nv;
}

// Samples BRDF, generating outgoing direction(Wo)
RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  // Get material params from input variable
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  // random variables
  float2 random = make_float2(rnd(seed), rnd(seed));
  pdf.matParams.u = random.x;
  pdf.matParams.v = random.y;

  // reflect I/origin on H to get omega_in/direction
  float3 H = GGX_Sample(pdf.view_direction, random, nu, nv);
  pdf.direction = reflect(pdf.view_direction, H);

  return pdf.direction;
}

// Gets BRDF PDF value
RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) {
  if (!Same_Hemisphere(pdf.direction, pdf.view_direction)) return 0.0f;

  // Get material params from input variable
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  // Handles degenerate cases for microfacet reflection
  float3 H = normalize(pdf.view_direction + pdf.direction);

  return GGX_PDF(H, pdf.view_direction, nu, nv) /
         (4.f * dot(pdf.view_direction, H));
}

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3 BRDF_Evaluate(PDFParams &pdf) {
  // Get material params from input variable
  float3 Rs = pdf.matParams.attenuation;
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  float3 origin = pdf.view_direction;
  float3 direction = normalize(pdf.direction);

  float cosThetaI = AbsCosTheta(origin);
  float cosThetaO = AbsCosTheta(direction);
  if (cosThetaI == 0 || cosThetaO == 0) return make_float3(0.f);

  float3 H = origin + direction;
  if (isNull(H)) return make_float3(0.f);

  H = normalize(H);
  float HdotI = dot(H, origin);  // origin or direction here

  float3 F = schlick(Rs, HdotI);  // Fresnel Reflectance
  float G = GGX_G(origin, direction, pdf.geometric_normal, nu,
                  nv);         // Geometric Shadowing
  float D = GGX_D(H, nu, nv);  // Normal Distribution Function(NDF)

  return Rs * D * G * F / (4.f * cosThetaI * cosThetaO);
}