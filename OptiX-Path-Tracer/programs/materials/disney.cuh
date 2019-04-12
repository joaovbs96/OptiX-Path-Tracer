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


