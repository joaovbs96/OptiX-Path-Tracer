#include "material.cuh"

//////////////////////////////////////////
// --- Oren-Nayar Reflectance Model --- //
//////////////////////////////////////////

// Original Paper: "Generalization of Lambert’s Reflectance Model"
// http://www.cs.columbia.edu/CAVE/projects/oren/
// http://www1.cs.columbia.edu/CAVE/publications/pdfs/Oren_SIGGRAPH94.pdf

// Yasuhiro Fujii’s "A tiny improvement of Oren-Nayar reflectance model" variant
// http://mimosa-pudica.net/improved-oren-nayar.html

// Reference Implementation:
// https://developer.blender.org/diffusion/C/browse/master/src/kernel/closure/bsdf_oren_nayar.h

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                 // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );             // ray PRD
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );  // from geometry

// Material Parameters
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  sample_texture, , );
rtDeclareVariable(float, rA, , );
rtDeclareVariable(float, rB, , );

///////////////////////////
// --- BRDF Programs --- //
///////////////////////////

// Assigns material and hit parameters to PRD
RT_PROGRAM void closest_hit() {
  prd.matType = Oren_Nayar_Material;
  prd.isSpecular = false;
  prd.scatterEvent = rayGotBounced;

  // Get hit params
  prd.origin = hit_rec.p;
  prd.geometric_normal = hit_rec.geometric_normal;
  prd.shading_normal = hit_rec.shading_normal;

  // Get material color
  int index = hit_rec.index;
  float3 color = sample_texture(hit_rec.u, hit_rec.v, hit_rec.p, index);

  // Assign material parameters to PRD, to be used in the sampling programs
  prd.matParams.orenNayar.rA = rA;
  prd.matParams.orenNayar.rB = rB;
  prd.matParams.attenuation = color;
}

// Samples BRDF, generating outgoing direction(Wo)
RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  float3 temp;
  cosine_sample_hemisphere(rnd(seed), rnd(seed), temp);

  Onb uvw(pdf.normal);
  uvw.inverse_transform(temp);

  pdf.direction = temp;

  return pdf.direction;
}

// Gets BRDF PDF value
RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) {
  float cosine = dot(unit_vector(pdf.direction), unit_vector(pdf.normal));

  if (cosine < 0.f)
    return 0.f;
  else
    return cosine / PI_F;
}

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3 BRDF_Evaluate(PDFParams &pdf) {
  if (dot(pdf.direction, pdf.normal) <= 0.f) return make_float3(0.f);

  float nl = max(dot(unit_vector(pdf.direction), unit_vector(pdf.normal)), 0.f);
  float nv = max(dot(unit_vector(pdf.origin), unit_vector(pdf.normal)), 0.f);
  float t = dot(unit_vector(pdf.direction), unit_vector(pdf.origin)) - nl * nv;

  if (t > 0.f) t /= max(nl, nv) + 0.001f;

  float rA = pdf.matParams.orenNayar.rA;
  float rB = pdf.matParams.orenNayar.rB;
  float3 color = pdf.matParams.attenuation;

  return color * (nl * (rA + rB * t));
}