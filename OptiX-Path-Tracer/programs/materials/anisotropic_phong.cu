
#include "material.cuh"
#include "microfacets.cuh"

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
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );  // from geometry

// Material Parameters
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  diffuse_color, , );
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  specular_color, , );
rtDeclareVariable(float, nu, , );
rtDeclareVariable(float, nv, , );

///////////////////////////
// --- BRDF Programs --- //
///////////////////////////

// Assigns material and hit parameters to PRD
RT_PROGRAM void closest_hit() {
  prd.matType = Anisotropic_Material;
  prd.isSpecular = false;
  prd.scatterEvent = rayGotBounced;

  // Get hit params
  prd.origin = hit_rec.p;
  prd.geometric_normal = hit_rec.geometric_normal;
  prd.shading_normal = hit_rec.shading_normal;

  // Get material colors
  int index = hit_rec.index;
  float3 diffuse = diffuse_color(hit_rec.u, hit_rec.v, hit_rec.p, index);
  float3 specular = specular_color(hit_rec.u, hit_rec.v, hit_rec.p, index);

  // Assign material parameters to PRD, to be used in the sampling programs
  prd.matParams.anisotropic.diffuse_color = diffuse;
  prd.matParams.anisotropic.specular_color = specular;
  prd.matParams.anisotropic.nu = nu;
  prd.matParams.anisotropic.nv = nv;
}

// Samples BRDF, generating outgoing direction(Wo)
RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  // Get material params from input variable
  float nu = Beckmann_Roughness(pdf.matParams.anisotropic.nu);
  float nv = Beckmann_Roughness(pdf.matParams.anisotropic.nv);

  // random variables
  float u = rnd(seed);
  float v = rnd(seed);

  float3 direction;
  if (u < 0.5) {
    u = ffmin(2 * u, 1.f - 0.001f);

    // Cosine-sample the hemisphere, flipping the direction if necessary
    cosine_sample_hemisphere(u, v, direction);

    Onb uvw(pdf.normal);
    uvw.inverse_transform(direction);
    if (pdf.origin.z < 0.f) direction.z *= -1.f;

  } else {
    u = ffmin(2 * (u - 0.5f), 1.f - 0.001f);

    // Sample microfacet orientation(H) and reflected direction(origin)
    float3 H = Beckmann_Sample(pdf.origin, u, v);
    direction = reflect(pdf.origin, H);  // on PBRT, wi is direction

    if (!SameHemisphere(pdf.origin, direction)) direction = make_float3(0.f);
  }

  pdf.direction = direction;

  return pdf.direction;
}

// Gets BRDF PDF value
RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) {
  if (!SameHemisphere(pdf.origin, pdf.direction)) return 0.f;

  // Get material params from input variable
  float nu = Beckmann_Roughness(pdf.matParams.anisotropic.nu);
  float nv = Beckmann_Roughness(pdf.matParams.anisotropic.nv);

  // half vector = (v1 + v2) / |v1 + v2|
  float3 H = normalize(pdf.direction + pdf.origin);
  float H_PDF = Beckmann_PDF(H, nu, nv);
  float HdotI = dot(H, pdf.origin);

  float AbsCosThetaWo = AbsCosTheta(pdf.direction);

  return 0.5f * (AbsCosThetaWo * (1.f / PI_F) + H_PDF / (4.f * HdotI));
}

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3 BRDF_Evaluate(PDFParams &pdf) {
  // Get material params from input variable
  float3 diffuse_color, specular_color;
  diffuse_color = pdf.matParams.anisotropic.diffuse_color;
  specular_color = pdf.matParams.anisotropic.specular_color;
  float nu = Beckmann_Roughness(pdf.matParams.anisotropic.nu);
  float nv = Beckmann_Roughness(pdf.matParams.anisotropic.nv);

  float nk1 = AbsCosTheta(pdf.origin);     // wi
  float nk2 = AbsCosTheta(pdf.direction);  // wo

  // diffuse component
  float3 diffuse_component = diffuse_color;
  diffuse_component *= (28.f / (23.f * PI_F));
  diffuse_component *= (make_float3(1.f) - specular_color);
  diffuse_component *= (1.f - fresnel_schlick(nk1 * 0.5f));
  diffuse_component *= (1.f - fresnel_schlick(nk2 * 0.5f));

  // half vector = (v1 + v2) / |v1 + v2|
  float3 H = normalize(pdf.direction + pdf.origin);
  if (isNull(H)) return make_float3(0.f);
  float HdotO = dot(H, pdf.direction);

  // specular component
  float3 specular_component = schlick(specular_color, HdotO);
  specular_component *= Beckmann_D(H, nu, nv);
  specular_component /= (4.f * abs(HdotO) * ffmax(nk1, nk2));

  return diffuse_component + specular_component;
}