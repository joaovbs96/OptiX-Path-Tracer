
#include "material.cuh"

// Paper & Tech Report
// https://www.cs.utah.edu/~shirley/papers/jgtbrdf.pdf
// https://www.cs.utah.edu/docs/techreports/2000/pdf/UUCS-00-014.pdf

// reference:
// https://github.com/JerryCao1985/SORT/blob/master/src/bsdf/ashikhmanshirley.cpp
// https://github.com/JerryCao1985/SORT/blob/master/src/bsdf/ashikhmanshirley.h

// origin -> k1
// direction -> k2
// N -> normal
// Rd -> diffuse color(of the 'substrate' under the specular coating)
// Rs -> specular color
// nu, nv -> phong parameters

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

// Material Programs
RT_PROGRAM void closest_hit() {
  prd.matType = Anisotropic_Material;
  prd.isSpecular = false; // this parameter refers to 'ideal specular BRDFs', like Metal and Dielectric
  prd.scatterEvent = rayGotBounced;

  prd.origin = hit_rec.p;
  prd.geometric_normal = hit_rec.geometric_normal;
  prd.shading_normal = hit_rec.shading_normal;

  // Get Material Colors
  int index = hit_rec.index;
  float3 diffuse = diffuse_color(hit_rec.u, hit_rec.v, hit_rec.p, index);
  float3 specular = specular_color(hit_rec.u, hit_rec.v, hit_rec.p, index);

  // Assign material parameters to PRD, to be used in the BRDF programs
  MaterialParameters params;
  params.u = hit_rec.u;
  params.v = hit_rec.v;
  params.anisotropic.nu = nu;
  params.anisotropic.nv = nv;
  params.anisotropic.diffuse_color = diffuse;
  params.anisotropic.specular_color = specular;
  prd.matParams = params;
}

RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  // Get material params from input variable
  MaterialParameters param = pdf.matParams;
  float nu = param.anisotropic.nu;
  float nv = param.anisotropic.nv;
  float u = rnd(seed);
  float v = rnd(seed);

  float3 direction;
  if (u < 0.5f) {
    cosine_sample_hemisphere(u, v, direction);

    Onb uvw(pdf.normal);
    uvw.inverse_transform(direction);
  } else {
    direction = Blinn_Sample(u, v, nu, nv);
    direction = 2 * dot(pdf.origin, direction) * direction - pdf.origin;
  }

  pdf.direction = direction;

  return pdf.direction;
}

// TODO: check microfacet BRDF description in the SIGGRAPH courses and the following link
// http://simonstechblog.blogspot.com/2011/12/microfacet-brdf.html

// TODO: check blinn functions, they are most likely returning the NaNs

RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) {
  // Get material params from input variable
  MaterialParameters param = pdf.matParams;
  float nu = param.anisotropic.nu;
  float nv = param.anisotropic.nv;

  // half vector = (v1 + v2) / |v1 + v2|
  float3 half_vector = unit_vector(pdf.origin + pdf.direction);
  //printf("%f %f %f \n", half_vector.x, half_vector.y, half_vector.z);
  float h_pdf = Blinn_PDF(half_vector, nu, nv);
  h_pdf = ffmax(0.001f, h_pdf);

  float a = AbsCosTheta(pdf.origin) / PI_F;
  float b = h_pdf / (4.f * dot(pdf.direction, half_vector));
  float t = 0.5f;

  // FIXME: it's returning NaN

  return ffmax(0.001f, lerp(a, b, t));
}

RT_CALLABLE_PROGRAM float3 BRDF_Evaluate(PDFParams &pdf) {
  // Get material params from input variable
  MaterialParameters param = pdf.matParams;
  float3 diffuse_color = param.anisotropic.diffuse_color;
  float3 specular_color = param.anisotropic.specular_color;
  float nu = param.anisotropic.nu;
  float nv = param.anisotropic.nv;

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
  float IoH = dot(pdf.origin, unit_vector(half_vector));
  printf("%f\n", IoH);
  float3 specular_component = schlick(specular_color, IoH);
  specular_component *= Blinn_Density(half_vector, nu, nv);
  specular_component /= 4.f * IoH * ffmax(nk1, nk2);

  // FIXME: Specular is returning NaN

  return (diffuse_component + specular_component) * nk1;
}