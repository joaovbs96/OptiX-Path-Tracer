
#include "material.cuh"
#include "microfacets.cuh"

////////////////////////////////////////////////////////////
// --- Ashikhmin-Shirley Anisotropic Phong BRDF Model --- //
////////////////////////////////////////////////////////////

// Original Paper & Tech Report - "An Anisotropic Phong Light Reflection Model"
// https://www.cs.utah.edu/~shirley/papers/jgtbrdf.pdf
// https://www.cs.utah.edu/docs/techreports/2000/pdf/UUCS-00-014.pdf

// Other references:
// https://github.com/JerryCao1985/SORT/blob/master/src/bsdf/ashikhmanshirley.cpp
// https://github.com/JerryCao1985/SORT/blob/master/src/bsdf/ashikhmanshirley.h

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
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  float u = rnd(seed);
  float v = rnd(seed);

  float3 direction;
  if (u < 0.5f) {
    cosine_sample_hemisphere(u, v, direction);

    Onb uvw((pdf.normal));
    uvw.inverse_transform(direction);

  } else {
    direction = Blinn_Sample(u, v, nu, nv);
    direction = 2 * dot(pdf.origin, direction) * direction - pdf.origin;
  }

  pdf.direction = direction;

  return pdf.direction;
}

// Gets BRDF PDF value
RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) {
  if (dot(pdf.direction, pdf.normal) <= 0.f) return 0.f;

  // Get material params from input variable
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  // half vector = (v1 + v2) / |v1 + v2|
  float3 half_vector = unit_vector(pdf.origin + pdf.direction);
  float h_pdf = Blinn_PDF(half_vector, nu, nv);

  float a = AbsCosTheta(pdf.origin) / PI_F;
  float b = h_pdf / (4.f * dot(pdf.direction, half_vector));
  float t = 0.5f;

  return lerp(a, b, t);
}

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3 BRDF_Evaluate(PDFParams &pdf) {
  if (dot(pdf.direction, pdf.normal) <= 0.f) return make_float3(0.f);

  // Get material params from input variable
  float3 diffuse_color, specular_color;
  diffuse_color = pdf.matParams.anisotropic.diffuse_color;
  specular_color = pdf.matParams.anisotropic.specular_color;
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  float nk1 = AbsCosTheta(pdf.origin);     // cos_theta_i
  float nk2 = AbsCosTheta(pdf.direction);  // cos_theta_o

  // diffuse component
  float3 diffuse_component = 28.f * diffuse_color;
  diffuse_component /= 23.f * PI_F;
  diffuse_component *= (make_float3(1.f) - specular_color);
  diffuse_component *= (1.f - fresnel_schlick(nk1 / 2));
  diffuse_component *= (1.f - fresnel_schlick(nk2 / 2));

  // half vector = (v1 + v2) / |v1 + v2|
  float3 half_vector = unit_vector(pdf.origin + pdf.direction);

  // specular component
  float IoH = dot(pdf.origin, half_vector);
  float3 specular_component = schlick(specular_color, IoH);
  specular_component *= Blinn_Density(half_vector, nu, nv);
  specular_component /= 4.f * IoH * ffmax(nk1, nk2);

  return (diffuse_component + specular_component) * nk1;
}