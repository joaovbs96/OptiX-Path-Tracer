#include "material.cuh"
#include "microfacets.cuh"

RT_FUNCTION void CalculateLobePdfs(const SurfaceParameters& surface,
                              float& pSpecular, float& pDiffuse, float& pClearcoat, float& pSpecTrans)
{
    float metallicBRDF   = surface.metallic;
    float specularBSDF   = (1.0f - surface.metallic) * surface.specTrans;
    float dielectricBRDF = (1.0f - surface.specTrans) * (1.0f - surface.metallic);

    float specularWeight     = metallicBRDF + dielectricBRDF;
    float transmissionWeight = specularBSDF;
    float diffuseWeight      = dielectricBRDF;
    float clearcoatWeight    = 1.0f * Saturate(surface.clearcoat); 

    float norm = 1.0f / (specularWeight + transmissionWeight + diffuseWeight + clearcoatWeight);

    pSpecular  = specularWeight     * norm;
    pSpecTrans = transmissionWeight * norm;
    pDiffuse   = diffuseWeight      * norm;
    pClearcoat = clearcoatWeight    * norm;
}

////////////////////////
// --- Sheen Lobe --- //
////////////////////////

RT_FUNCTION float3 CalculateTint(const float3& baseColor) {
  float luminance = dot(make_float3(0.3f, 0.6f, 1.0f), baseColor);
  return luminance > 0.0f ? baseColor * (1.0f / luminance) : make_float3(1.f);
}

RT_FUNCTION float3 EvaluateSheen(const Disney_Parameters& surface,
                                 const float3& Wo,    // view direction
                                 const float3& H,     // half vector
                                 const float3& Wi) {  // light direction
  if (surface.sheen <= 0.0f) return make_float3(0.f);

  float3 tint = CalculateTint(surface.baseColor);
  float3 sheenColor = lerp(make_float3(1.f), tint, surface.sheenTint);
  return SchlickWeight(dot(H, Wi)) * sheenColor * surface.sheen;
}

////////////////////////////
// --- Clearcoat Lobe --- //
////////////////////////////

// Burley's Generalized Trowbridge-Reitz
RT_FUNCTION float GTR1(float absDotHL, float a) {
  if (a >= 1) return PI_F;

  float a2 = Square(a);
  float absDotHL2 = Square(absDotHL);
  return (a2 - 1.0f) / (PI_F * log2f(a2) * (1.0f + (a2 - 1.0f) * absDotHL2));
}

RT_FUNCTION float EvaluateDisneyClearcoat(
    const Disney_Parameters& surface,
    const float3& Wo,    // view direction
    const float3& H,     // half vector
    const float3& Wi) {  // light direction
  if (surface.clearcoat <= 0.f) return 0.f;

  float absDotNH = AbsCosTheta(H);
  float absDotNL = AbsCosTheta(Wi);
  float absDotNV = AbsCosTheta(Wo);
  float dotHL = dot(H, Wi);

  float d = GTR1(absDotNH, lerp(0.1f, 0.001f, surface.clearcoatGloss));
  float f = schlick(0.04f, dotHL);
  float gl = GGX_G1(Wi, 0.25f);
  float gv = GGX_G1(Wo, 0.25f);

  return 0.25f * surface.clearcoat * d * f * gl * gv;
}

///////////////////////////
// --- Specular BRDF --- //
///////////////////////////

// TODO: check if common GGX_D can be used here
RT_FUNCTION float Disney_GGX_D(const float3& H, float ax, float ay) {
  float dotHX2 = Square(H.x);
  float dotHY2 = Square(H.z);
  float cos2Theta = Cos2Theta(H);
  float ax2 = Square(ax);
  float ay2 = Square(ay);
  float beta2 = Square(dotHX2 / ax2 + dotHY2 / ay2 + cos2Theta);

  return 1.f / (PI_F * ax * ay * beta2);
}

RT_FUNCTION float Disney_GGX_G1(const float3& W,  // view or light direction
                                const float3& H,  // half vector
                                float ax, float ay) {
  float dotHW = dot(W, H);
  if (dotHW <= 0.f) return 0.f;

  float absTanTheta = fabsf(TanTheta(W));
  if (isinf(absTanTheta)) return 0.f;

  float a = sqrtf(Cos2Phi(W) * ax * ax + Sin2Phi(W) * ay * ay);
  float a2Tan2Theta = Square(a * absTanTheta);

  float lambda = 0.5f * (-1.f + sqrtf(1.f + a2Tan2Theta));
  return 1.f / (1.f + lambda);
}

RT_FUNCTION void Disney_Anisotropic_Params(const float roughness,
                                           const float anisotropic, float& ax,
                                           float& ay) {
  float aspect = sqrtf(1.f - 0.9f * anisotropic);
  ax = fmaxf(0.001f, Square(roughness) / aspect);
  ay = fmaxf(0.001f, Square(roughness) * aspect);
}

RT_FUNCTION float3 Disney_Fresnel(const Disney_Parameters& surface,
                                  const float3& Wo,    // view direction
                                  const float3& H,     // half vector
                                  const float3& Wi) {  // light direction
  float dotHV = dot(H, Wo);

  float3 tint = CalculateTint(surface.baseColor);

  // See section 3.1 and 3.2 of the 2015 PBR presentation + the Disney BRDF
  // explorer (which does their 2012 remapping rather than the
  // SchlickR0FromRelativeIOR seen here but they mentioned the switch in 3.2).
  float3 R0 = lerp(make_float3(1.f), tint, surface.specularTint);
  R0 *= SchlickR0FromRelativeIOR(surface.relativeIOR);
  R0 = lerp(R0, surface.baseColor, surface.metallic);

  float fresnel = Fresnel_Dielectric(dotHV, 1.0f, surface.ior);
  float3 metallicFresnel = schlick(R0, dot(Wi, H));
  float3 dielectricFresnel = make_float3(fresnel);

  return lerp(dielectricFresnel, metallicFresnel, surface.metallic);
}

RT_FUNCTION float3 EvaluateDisneyBRDF(const Disney_Parameters& surface,
                                      const float3& Wo,    // view direction
                                      const float3& H,     // half vector
                                      const float3& Wi) {  // light direction
  float dotNL = CosTheta(Wi);
  float dotNV = CosTheta(Wo);
  if (dotNL <= 0.0f || dotNV <= 0.0f) return make_float3(0.f);

  float ax, ay;
  Disney_Anisotropic_Params(surface.roughness, surface.anisotropic, ax, ay);

  float d = Disney_GGX_D(H, ax, ay);
  float gl = Disney_GGX_G1(Wi, H, ax, ay);
  float gv = Disney_GGX_G1(Wo, H, ax, ay);

  float3 f = Disney_Fresnel(surface, Wo, H, Wi);

  // TODO: correctly implement the PDF functions
  // Bsdf::GgxVndfAnisotropicPdf(Wi, H, Wo, ax, ay /*, fPdf, rPdf*/);

  return d * gl * gv * f / (4.0f * dotNL * dotNV);
}

///////////////////////////
// --- Specular BSDF --- //
///////////////////////////

RT_FUNCTION float ThinTransmissionRoughness(float ior, float roughness) {
  // Disney scales by (.65 * eta - .35) based on figure 15 of the 2015 PBR
  // course notes. Based on their figure the results match a geometrically thin
  // solid fairly well.
  return saturate((0.65f * ior - 0.35f) * roughness);
}

RT_FUNCTION float3
EvaluateDisneySpecTransmission(const Disney_Parameters& surface,
                               const float3& Wo,  // view direction
                               const float3& H,   // half vector
                               const float3& Wi,  // light direction
                               float ax, float ay, bool thin) {
  float relativeIor = surface.relativeIOR;
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

  float f = Fresnel_Dielectric(dotHV, 1.0f, 1.0f / surface.relativeIOR);

  float3 color;
  if (thin)
    color = sqrt(surface.baseColor);
  else
    color = surface.baseColor;

  // Note that we are intentionally leaving out the 1/n2 spreading factor since
  // for VCM we will be evaluating particles with this. That means we'll need to
  // model the air-[other medium] transmission if we ever place the camera
  // inside a non-air medium.
  float c = (absDotHL * absDotHV) / (absDotNL * absDotNV);
  float t = (n2 / Square(dotHL + relativeIor * dotHV));
  return color * c * t * (1.0f - f) * gl * gv * d;
}

//////////////////////////
// --- Diffuse BRDF --- //
//////////////////////////

RT_FUNCTION float EvaluateDisneyRetroDiffuse(
    const Disney_Parameters& surface,
    const float3& Wo,    // view direction
    const float3& H,     // half vector
    const float3& Wi) {  // light direction
  float dotNL = AbsCosTheta(Wi);
  float dotNV = AbsCosTheta(Wo);

  float roughness = surface.roughness * surface.roughness;

  float rr = 0.5f + 2.f * dotNL * dotNL * roughness;
  float fl = SchlickWeight(dotNL);
  float fv = SchlickWeight(dotNV);

  return rr * (fl + fv + fl * fv * (rr - 1.0f));
}

RT_FUNCTION float EvaluateDisneyDiffuse(const Disney_Parameters& surface,
                                        const float3& Wo,  // view direction
                                        const float3& H,   // half vector
                                        const float3& Wi,  // light direction
                                        bool thin) {
  float dotNL = AbsCosTheta(Wi);
  float dotNV = AbsCosTheta(Wo);

  float fl = SchlickWeight(dotNL);
  float fv = SchlickWeight(dotNV);

  float hKrueger = 0.f;

  if (thin && surface.flatness > 0.f) {
    float roughness = surface.roughness * surface.roughness;

    float dotHL = dot(H, Wi);
    float fss90 = dotHL * dotHL * roughness;
    float fss = lerp(1.0f, fss90, fl) * lerp(1.0f, fss90, fv);

    float ss = 1.25f * (fss * (1.0f / (dotNL + dotNV) - 0.5f) + 0.5f);
    hKrueger = ss;
  }

  float lambert = 1.f;
  float retro = EvaluateDisneyRetroDiffuse(surface, Wo, H, Wi);
  float subsurface = lerp(lambert, hKrueger, thin ? surface.flatness : 0.f);

  return INV_PI * (retro + subsurface * (1.f - 0.5f * fl) * (1.f - 0.5f * fv));
}