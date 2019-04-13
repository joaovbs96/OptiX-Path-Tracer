#include "material.cuh"
#include "microfacets.cuh"

/*

    struct SurfaceParameters
    {
        float3 position;
        float3x3 worldToTangent;
        float error;

        float3 view;
        
        float3 baseColor;
        float3 transmittanceColor;
        float sheen;
        float sheenTint;
        float clearcoat;
        float clearcoatGloss;
        float metallic;
        float specTrans;
        float diffTrans;
        float flatness;
        float anisotropic;
        float relativeIOR;
        float specularTint;
        float roughness;
        float scatterDistance;

        float ior;

        // -- material layer info
        ShaderType shader;
        uint32 materialFlags;

        uint32 lightSetIndex;
};

*/

////////////////////////
// --- Sheen Lobe --- //
////////////////////////

RT_FUNCTION float3 CalculateTint(const float3& baseColor) {
  float luminance = dot(make_float3(0.3f, 0.6f, 1.0f), baseColor);
  return luminance > 0.0f ? baseColor * (1.0f / luminance) : make_float3(1.f);
}

RT_FUNCTION float3 EvaluateSheen(const float sheen, const float sheenTint,
                                 const float3& baseColor, const float3& Wo,
                                 const float3& Wm, const float3& Wi) {
  if (sheen <= 0.0f) return make_float3(0.f);

  float3 tint = CalculateTint(baseColor);
  float3 sheenColor = sheen * lerp(make_float3(1.f), tint, sheenTint);
  return SchlickWeight(dot(Wm, Wi)) * sheenColor;
}

////////////////////////////
// --- Clearcoat Lobe --- //
////////////////////////////

// Burley's Generalized Trowbridge-Reitz
RT_FUNCTION float GTR1(float absDotHL, float a) {
  if (a >= 1) return PI_F;

  float a2 = a * a;
  return (a2 - 1.0f) /
         (PI_F * log2f(a2) * (1.0f + (a2 - 1.0f) * absDotHL * absDotHL));
}

RT_FUNCTION float EvaluateDisneyClearcoat(
    float clearcoat,     // clearcoat value
    float alpha,         // alpha
    const float3& Wo,    // view direction
    const float3& Wm,    // half vector
    const float3& Wi) {  // light direction
  if (clearcoat <= 0.0f) return 0.0f;

  float absDotNH = AbsCosTheta(Wm);
  float absDotNL = AbsCosTheta(Wi);
  float absDotNV = AbsCosTheta(Wo);
  float dotHL = dot(Wm, Wi);

  float d = GTR1(absDotNH, lerp(0.1f, 0.001f, alpha));
  float f = schlick(0.04f, dotHL);
  float gl = GGX_G1(Wi, 0.25f);
  float gv = GGX_G1(Wo, 0.25f);

  return 0.25f * clearcoat * d * f * gl * gv;
}

///////////////////////////
// --- Specular BRDF --- //
///////////////////////////

RT_FUNCTION float Disney_GGX_D(const float3& H, float ax, float ay) {
  float dotHX2 = sqrf(H.x);
  float dotHY2 = sqrf(H.z);
  float cos2Theta = Cos2Theta(H);
  float ax2 = sqrf(ax);
  float ay2 = sqrf(ay);

  return 1.f / (PI_F * ax * ay * sqrf(dotHX2 / ax2 + dotHY2 / ay2 + cos2Theta));
}

RT_FUNCTION float Disney_GGX_G1(const float3& w, const float3& wm, float ax, float ay) {
  float dotHW = dot(w, wm);
  if (dotHW <= 0.f) return 0.f;

  float absTanTheta = fabsf(TanTheta(w));
  if(IsInf(absTanTheta)) return 0.f;

  float a = sqrtf(Cos2Phi(w) * ax * ax + Sin2Phi(w) * ay * ay);
  float a2Tan2Theta = sqrf(a * absTanTheta);

  float lambda = 0.5f * (-1.f + sqrtf(1.f + a2Tan2Theta));
  return 1.f / (1.f + lambda);
}

RT_FUNCTION float3 EvaluateDisneyBRDF(const SurfaceParameters& surface, 
                                      const float3& Wo, 
                                      const float3& H, 
                                      const float3& Wi,
                                      const float roughness,
                                      const float anisotropic) {
  float dotNL = CosTheta(Wi);
  float dotNV = CosTheta(Wo);
  if(dotNL <= 0.0f || dotNV <= 0.0f) return make_float3(0.f);

  float ax, ay;
  CalculateAnisotropicParams(roughness, anisotropic, ax, ay);

  float d = Disney_GGX_D(H, ax, ay);
  float gl = Disney_GGX_G1(Wi, H, ax, ay);
  float gv = Disney_GGX_G1(Wo, H, ax, ay);

  float3 f = Disney_Fresnel(surface, Wo, H, Wi);

  // TODO: correct this function call
  Bsdf::GgxVndfAnisotropicPdf(Wi, H, Wo, ax, ay/*, fPdf, rPdf*/);

  return d * gl * gv * f / (4.0f * dotNL * dotNV);
}

///////////////////////////
// --- Specular BSDF --- //
///////////////////////////

RT_FUNCTION float ThinTransmissionRoughness(float ior, float roughness) {
  // Disney scales by (.65 * eta - .35) based on figure 15 of the 2015 PBR course notes. 
  // Based on their figure the results match a geometrically thin solid fairly well.
  return saturate((0.65f * ior - 0.35f) * roughness);
}

RT_FUNCTION float3 EvaluateDisneySpecTransmission(const float3& Wo, 
                                                  const float3& H, 
                                                  const float3& Wi, 
                                                  float ax, 
                                                  float ay, 
                                                  bool thin,
                                                  const float3& baseColor,
                                                  const float relativeIOR) {
  float relativeIor = relativeIOR;
  float n2 = relativeIor * relativeIor;

  float absDotNL = AbsCosTheta(Wi);
  float absDotNV = AbsCosTheta(Wo);
  float dotHL = dot(H, Wi);
  float dotHV = dot(H, Wo);
  float absDotHL = fabsf(dotHL);
  float absDotHV = fabsf(dotHV);

  float d = Disney_GGX_D(H, ax, ay);
  float gl = Disney_GGX_G1(Wi, H, ax, ay);
  float gv = Disney_GGX_G1(Wo, H, ax, ay);

  float f = Fresnel::Dielectric(dotHV, 1.0f, 1.0f / relativeIOR);

  float3 color;
  if(thin)
    color = sqrt(baseColor);
  else
    color = baseColor;

  // Note that we are intentionally leaving out the 1/n2 spreading factor since for VCM we will be evaluating
  // particles with this. That means we'll need to model the air-[other medium] transmission if we ever place
  // the camera inside a non-air medium.
  float c = (absDotHL * absDotHV) / (absDotNL * absDotNV);
  float t = (n2 / sqrf(dotHL + relativeIor * dotHV));
  return color * c * t * (1.0f - f) * gl * gv * d;
}

//////////////////////////
// --- Diffuse BRDF --- //
//////////////////////////

RT_FUNCTION float EvaluateDisneyDiffuse(const float3& Wo, 
                                        const float3& H, 
                                        const float3& Wi, 
                                        bool thin,
                                        const float roughness,
                                        const float flatness) {
  float dotNL = AbsCosTheta(wi);
  float dotNV = AbsCosTheta(wo);

  float fl = SchlickWeight(dotNL);
  float fv = SchlickWeight(dotNV);

  float hanrahanKrueger = 0.f;

  if(thin && flatness > 0.f) {
    float roughness = roughness * roughness;

    float dotHL = dot(wm, wi);
    float fss90 = dotHL * dotHL * roughness;
    float fss = lerp(1.0f, fss90, fl) * lerp(1.0f, fss90, fv);

    float ss = 1.25f * (fss * (1.f / (dotNL + dotNV) - 0.5f) + 0.5f);
    hanrahanKrueger = ss;
  }

  float lambert = 1.f;
  float retro = EvaluateDisneyRetroDiffuse(surface, wo, wm, wi);
  float subsurfaceApprox = lerp(lambert, hanrahanKrueger, thin ? flatness : 0.f);

  return (1.f / PI_F) * (retro + subsurfaceApprox * (1.f - 0.5f * fl) * (1.f - 0.5f * fv));
}