#include "disney.cuh"

////////////////////////////////////
// --- Disney Principled BSDF --- //
////////////////////////////////////

// Based on blog post and code by Joe Schutte
// https://schuttejoe.github.io/post/disneybsdf/
// https://github.com/schuttejoe/Selas/blob/dev/Source/Core/Shading/Disney.cpp
// https://github.com/schuttejoe/Selas/blob/dev/Source/Core/Shading/Disney.h

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                 // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );             // ray PRD
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );  // from geometry

// Material Parameters
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  BaseColor, , );
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  TransmittanceColor, , );
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
RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  // Get material params from input variable
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  float3 Wo = pdf.view_direction;  // outgoing, to camera

  // create basis
  float3 N = normalize(pdf.geometric_normal);
  float3 T = normalize(cross(N, make_float3(0.f, 1.f, 0.f)));
  float3 B = cross(T, N);

  // random variables
  float2 random = make_float2(rnd(seed), rnd(seed));
  pdf.matParams.u = random.x;
  pdf.matParams.v = random.y;

  // get half vector and rotate it to world space
  float3 H = normalize(GGX_Sample(Wo, random, nu, nv));
  H = H.x * B + H.y * N + H.z * T;

  float HdotI = dot(H, Wo);
  if (HdotI < 0.f) H = -H;

  float3 Wi = normalize(-Wo + 2.f * dot(Wo, H) * H);  // reflect(Wo, H)

  pdf.direction = Wi;

  return pdf.direction;
}

// Gets BRDF PDF value
RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) {
  float3 Wo = pdf.view_direction;
  float3 Wi = pdf.localDirection;

  // Get material params from input variable
  float nu = pdf.matParams.anisotropic.nu;
  float nv = pdf.matParams.anisotropic.nv;

  // Handles degenerate cases for microfacet reflection
  float3 H = normalize(Wi + Wo);

  return GGX_PDF(H, Wo, nu, nv) / (4.f * dot(Wo, H));
}

// Evaluates BRDF, returning its reflectance
RT_CALLABLE_PROGRAM float3 BRDF_Evaluate(PDFParams &pdf) {
  float3 Wo = pdf.view_direction, Wi = pdf.direction;
  // float3 wo = Normalize(MatrixMultiply(v, surface.worldToTangent));
  // float3 wi = Normalize(MatrixMultiply(l, surface.worldToTangent));
  float3 H = normalize(Wo + Wi);

  // Get material params from input variable
  Disney_Parameters surface = pdf.matParams.disney;

  float dotNV = CosTheta(Wo);
  float dotNL = CosTheta(Wi);

  float3 reflectance = make_float3(0.f);

  // TODO: implement pdf functions

  float3 baseColor = surface.baseColor;
  float metallic = surface.metallic;
  float specTrans = surface.specTrans;
  float roughness = surface.roughness;

  // calculate all of the anisotropic params
  float ax, ay;
  Disney_Anisotropic_Params(surface.roughness, surface.anisotropic, ax, ay);

  float diffuseWeight = (1.f - metallic) * (1.f - specTrans);
  float transWeight = (1.f - metallic) * specTrans;

  // Clearcoat
  bool upperHemisphere = dotNL > 0.f && dotNV > 0.f;
  if (upperHemisphere && surface.clearcoat > 0.f) {
    float clearcoat = EvaluateDisneyClearcoat(surface, Wo, H, Wi);
    reflectance += float3(clearcoat);
  }

  // Diffuse
  if (diffuseWeight > 0.f) {
    float diffuse = EvaluateDisneyDiffuse(surface, Wo, H, Wi, thin);
    float3 sheen = EvaluateSheen(surface, Wo, H, Wi);

    reflectance += diffuseWeight * (diffuse * surface.baseColor + sheen);
  }

  // Transmission
  if (transWeight > 0.f) {
    // Scale roughness based on IOR (Burley 2015, Figure 15).
    float rscaled =
        thin ? ThinTransmissionRoughness(surface.ior, surface.roughness)
             : surface.roughness;
    float tax, tay;
    Disney_Anisotropic_Params(rscaled, surface.anisotropic, tax, tay);

    float3 transmission =
        EvaluateDisneySpecTransmission(surface, Wo, H, Wi, tax, tay, thin);
    reflectance += transWeight * transmission;
  }

  // -- specular
  if (upperHemisphere) {
    float3 specular = EvaluateDisneyBRDF(
        surface, wo, wm, wi, forwardMetallicPdfW, reverseMetallicPdfW);

    reflectance += specular;
  }

  reflectance = reflectance * Absf(dotNL);

  return reflectance;
}