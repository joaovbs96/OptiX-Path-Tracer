
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

// Blender Parameters
// I - origin - omega_out
// O - direction - omega_in

///////////////////////////
// --- BRDF Programs --- //
///////////////////////////

// Assigns material and hit parameters to PRD
RT_PROGRAM void closest_hit() {
  prd.matType = Anisotropic_BRDF;
  prd.isSpecular = false;
  prd.scatterEvent = rayGotBounced;

  // Get hit params
  prd.origin = hit_rec.p;
  prd.geometric_normal = hit_rec.geometric_normal;
  prd.shading_normal = hit_rec.shading_normal;
  prd.view_direction = hit_rec.view_direction;

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

  // random variables
  float2 random = make_float2(rnd(seed), rnd(seed));
  pdf.matParams.u = random.x;
  pdf.matParams.v = random.y;

  float3 direction, origin = pdf.view_direction;
  if (random.x < 0.5) {
    // sample diffuse term
    random.x = fminf(2 * random.x, 1.f - 1e-6f);

    // Cosine-sample the hemisphere
    cosine_sample_hemisphere(random.x, random.y, direction);

    Onb uvw(normalize(pdf.normal));
    uvw.inverse_transform(direction);

    // flip the direction if necessary
    if (!Same_Hemisphere(direction, origin)) direction *= -1;

  } else {
    // sample specular term
    random.x = fminf(2 * (random.x - 0.5f), 1.f - 1e-6f);

    // reflect I/origin on H to get omega_in/direction
    float3 H = GGX_Sample(origin, random, nu, nv);
    direction = reflect(origin, H);
  }

  pdf.direction = direction;

  return pdf.direction;
}

// Gets BRDF PDF value
RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) {
  if (!Same_Hemisphere(pdf.direction, pdf.view_direction)) return 0.0f;

  // Get material params from input variable
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  float3 origin = pdf.view_direction;
  float3 direction = normalize(pdf.direction);

  // PDF Diffuse Term
  float diffuse_PDF = dot(direction, normalize(pdf.normal));
  if (diffuse_PDF < 0.f) diffuse_PDF = 0.f;

  // PDF Specular Term
  const float3 H = normalize(origin + direction);
  const float specular_PDF =
      GGX_PDF(H, origin, nu, nv) / (4.f * dot(origin, H));

  return (diffuse_PDF + specular_PDF) / 2.f;
}

// TODO: attempt to implement the Blender one once again
// https://developer.blender.org/diffusion/C/browse/master/src/kernel/closure/bsdf_ashikhmin_shirley.h;f403fe5d73b6eaae955df89191b452b424215a6b?as=source&blame=off

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3 BRDF_Evaluate(PDFParams &pdf) {
  if (!Same_Hemisphere(pdf.direction, pdf.view_direction))
    return make_float3(0.f);

  // Get material params from input variable
  float3 Rd = pdf.matParams.anisotropic.diffuse_color;
  float3 Rs = pdf.matParams.anisotropic.specular_color;
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  float3 origin = pdf.view_direction;
  float3 direction = normalize(pdf.direction);

  float nk1 = AbsCosTheta(direction);
  float nk2 = AbsCosTheta(origin);
  if (nk1 == 0 || nk2 == 0) return make_float3(0.f);

  // half vector = (v1 + v2) / |v1 + v2|
  float3 H = origin + direction;
  if (isNull(H)) return make_float3(0.f);

  // diffuse component
  float3 diffuse_component = Rd * (28.f / (23.f * PI_F));
  diffuse_component *= (make_float3(1.f) - Rs);
  diffuse_component *= (1.f - powf(1.f - nk1 * 0.5f, 5.f));
  diffuse_component *= (1.f - powf(1.f - nk2 * 0.5f, 5.f));

  H = normalize(H);
  float HdotI = dot(H, origin);  // origin or direction here

  // specular component
  float3 F = schlick(Rs, HdotI);  // fresnel reflectance term
  float D = GGX_D(H, nu, nv);     // Normal Distribution Function(NDF)
  float3 specular_component = F * D / (4.f * fabsf(HdotI) * fmaxf(nk1, nk2));

  return (diffuse_component + specular_component);
}