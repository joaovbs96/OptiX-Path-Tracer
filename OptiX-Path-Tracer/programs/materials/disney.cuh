#ifndef DISNEYCUH
#define DISNEYCUH

#include "../math/trigonometric.cuh"
#include "../pdfs/pdf.cuh"
#include "material.cuh"

RT_FUNCTION void CalculateLobePdfs(const Disney_Parameters& surface,
                                   float& pSpecular, float& pDiffuse,
                                   float& pClearcoat, float& pSpecTrans) {
  float metallicBRDF = surface.metallic;
  float specularBSDF = (1.0f - surface.metallic) * surface.specTrans;
  float dielectricBRDF = (1.0f - surface.specTrans) * (1.0f - surface.metallic);

  float specularWeight = metallicBRDF + dielectricBRDF;
  float transmissionWeight = specularBSDF;
  float diffuseWeight = dielectricBRDF;
  float clearcoatWeight = 1.0f * saturate(surface.clearcoat);

  float norm = 1.0f / (specularWeight + transmissionWeight + diffuseWeight +
                       clearcoatWeight);

  pSpecular = specularWeight * norm;
  pSpecTrans = transmissionWeight * norm;
  pDiffuse = diffuseWeight * norm;
  pClearcoat = clearcoatWeight * norm;
}

////////////////////////
// --- Sheen Lobe --- //
////////////////////////

RT_FUNCTION float3 CalculateTint(const float3& baseColor) {
  float luminance = dot(make_float3(0.3f, 0.6f, 1.0f), baseColor);
  return luminance > 0.0f ? baseColor * (1.0f / luminance) : make_float3(1.f);
}

RT_FUNCTION float3 Evaluate_Sheen(const Disney_Parameters& surface,
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

RT_FUNCTION float GGX_G1(const float3& V, float a) {
  float a2 = a * a;
  float absDotNV = AbsCosTheta(V);

  return 2.0f / (1.0f + sqrtf(a2 + (1 - a2) * absDotNV * absDotNV));
}

RT_FUNCTION float Evaluate_Clearcoat(const Disney_Parameters& surface,
                                     const float3& Wo,  // view direction
                                     const float3& H,   // half vector
                                     const float3& Wi,  // light direction
                                     float& pdf) {
  if (surface.clearcoat <= 0.f) return 0.f;

  float absDotNH = AbsCosTheta(H);
  float absDotNL = AbsCosTheta(Wi);
  float absDotNV = AbsCosTheta(Wo);
  float dotHL = dot(H, Wi);

  float d = GTR1(absDotNH, lerp(0.1f, 0.001f, surface.clearcoatGloss));
  float f = schlick(0.04f, dotHL);
  float gl = GGX_G1(Wi, 0.25f);
  float gv = GGX_G1(Wo, 0.25f);

  pdf = d / (4.0f * absDotNL);

  return 0.25f * surface.clearcoat * d * f * gl * gv;
}

///////////////////////////
// --- Specular BRDF --- //
///////////////////////////

RT_FUNCTION float GGX_D(const float3& H, float ax, float ay) {
  float dotHX2 = Square(H.x);
  float dotHY2 = Square(H.z);
  float cos2Theta = Cos2Theta(H);
  float ax2 = Square(ax);
  float ay2 = Square(ay);
  float beta2 = Square(dotHX2 / ax2 + dotHY2 / ay2 + cos2Theta);

  return 1.f / (PI_F * ax * ay * beta2);
}

RT_FUNCTION float GGX_G1(const float3& W,  // view or light direction
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

RT_FUNCTION float GGX_PDF(const float3& Wo,  // view direction
                          const float3& H,   // half vector
                          const float3& Wi,  // light direction
                          const float ax, const float ay) {
  float D = GGX_D(H, ax, ay);

  float absDotNL = AbsCosTheta(Wi);
  float absDotHL = fabsf(dot(H, Wi));
  float G1 = GGX_G1(Wo, H, ax, ay);
  return G1 * absDotHL * D / absDotNL;
}

RT_FUNCTION void Anisotropic_Params(const float roughness,
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

RT_FUNCTION float3 Evaluate_Specular(const Disney_Parameters& surface,
                                     const float3& Wo,  // view direction
                                     const float3& H,   // half vector
                                     const float3& Wi,  // light direction
                                     float& pdf) {
  float dotNL = CosTheta(Wi);
  float dotNV = CosTheta(Wo);
  if (dotNL <= 0.0f || dotNV <= 0.0f) return make_float3(0.f);

  float ax, ay;
  Anisotropic_Params(surface.roughness, surface.anisotropic, ax, ay);

  float d = GGX_D(H, ax, ay);
  float gl = GGX_G1(Wi, H, ax, ay);
  float gv = GGX_G1(Wo, H, ax, ay);

  float3 f = Disney_Fresnel(surface, Wo, H, Wi);

  pdf = GGX_PDF(Wi, H, Wo, ax, ay);

  return d * gl * gv * f / (4.0f * dotNL * dotNV);
}

///////////////////////////
// --- Specular BSDF --- //
///////////////////////////

RT_FUNCTION float Transmission_Roughness(float ior, float roughness) {
  // Disney scales by (.65 * eta - .35) based on figure 15 of the 2015 PBR
  // course notes. Based on their figure the results match a geometrically thin
  // solid fairly well.
  return saturate((0.65f * ior - 0.35f) * roughness);
}

// Sampling a normal respect to the NDF(PBRT 8.4.3)
RT_FUNCTION float3 GGX_Sample(float3 origin, float2 random, float nu,
                              float nv) {
  bool flip = origin.y < 0;

  // 1. stretch the view so we are sampling as though roughness==1
  float3 stretchedOrigin = make_float3(origin.x * nu, origin.y, origin.z * nv);
  stretchedOrigin = normalize(stretchedOrigin);

  // 2. simulate P22_{wi}(slopeX, slopeY, 1, 1)
  float slopeX, slopeY;
  float cosTheta = CosTheta(stretchedOrigin);

  // special case (normal incidence)
  if (cosTheta > 0.9999f) {
    float r = sqrtf(random.x / (1 - random.x));
    float phi = 6.28318530718 * random.y;
    slopeX = r * cos(phi);
    slopeY = r * sin(phi);
  } else {
    float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
    float tanTheta = sinTheta / cosTheta;
    float a = 1.f / tanTheta;
    float G1 = 2.f / (1.f + sqrtf(1.f + 1.f / (a * a)));

    // sample slope_x
    float A = 2.f * random.x / G1 - 1.f;
    float tmp = 1.f / (A * A - 1.f);
    if (tmp > 1e10f) tmp = 1e10f;
    float B = tanTheta;
    float D = sqrtf(fmaxf(B * B * tmp * tmp - (A * A - B * B) * tmp, 0.f));
    float slopeX1 = B * tmp - D;
    float slopeX2 = B * tmp + D;
    slopeX = (A < 0 || slopeX2 > 1.f / tanTheta) ? slopeX1 : slopeX2;

    // sample slope_y
    float S;
    if (random.y > 0.5f) {
      S = 1.f;
      random.y = 2.f * (random.y - .5f);
    } else {
      S = -1.f;
      random.y = 2.f * (.5f - random.y);
    }
    float z =
        (random.y * (random.y * (random.y * 0.27385f - 0.73369f) + 0.46341f)) /
        (random.y * (random.y * (random.y * 0.093073f + 0.309420f) - 1.f) +
         0.597999f);
    slopeY = S * z * sqrtf(1.f + slopeX * slopeX);
  }

  // 3. rotate
  float t = CosPhi(stretchedOrigin) * slopeX - SinPhi(stretchedOrigin) * slopeY;
  slopeY = SinPhi(stretchedOrigin) * slopeX + CosPhi(stretchedOrigin) * slopeY;
  slopeX = t;

  // 4. unstretch
  slopeX = nu * slopeX;
  slopeY = nv * slopeY;

  // 5. compute normal
  float3 H = normalize(make_float3(-slopeX, 1.f, -slopeY));

  if (flip) H *= -1;

  return H;
}

RT_FUNCTION float3 Calculate_Extinction(float3 apparantColor,
                                        float scatterDistance) {
  float3 a = apparantColor;
  float3 s = make_float3(1.9f) - a +
             3.5f * (a - make_float3(0.8f)) * (a - make_float3(0.8f));

  return 1.0f / (s * scatterDistance);
}

RT_FUNCTION float3 Sample_Transmission(const Disney_Parameters& surface,
                                       const float3& Wo,  // view direction
                                       bool thin, uint& seed) {
  // TODO: check worldToTangent transform
  // float3 wo = MatrixMultiply(v, surface.worldToTangent);
  float3 Wi;
  if (CosTheta(Wo) == 0.0) return make_float3(0.f);

  float rscaled;
  if (thin)  // Scale roughness based on IOR (Burley 2015, Figure 15).
    rscaled = Transmission_Roughness(surface.ior, surface.roughness);
  else
    rscaled = surface.roughness;

  float tax, tay;
  Anisotropic_Params(rscaled, surface.anisotropic, tax, tay);

  // -- Sample visible distribution of normals
  float2 random = make_float2(rnd(seed), rnd(seed));
  float3 H = GGX_Sample(Wo, random, tax, tay);

  float dotVH = dot(Wo, H);
  if (H.y < 0.0f) dotVH = -dotVH;

  float ni = Wo.y > 0.0f ? 1.0f : surface.ior;
  float nt = Wo.y > 0.0f ? surface.ior : 1.0f;
  float relativeIOR = ni / nt;

  // -- Disney uses the full dielectric Fresnel equation for transmission. We
  // also importance sample F
  // -- to switch between refraction and reflection at glancing angles.
  float F = Fresnel_Dielectric(dotVH, 1.0f, surface.ior);

  // Since we're sampling the distribution of visible normals the pdf cancels
  // out with a number of other terms. We are left with the weight
  // G2(wi, wo, wm) / G1(wi, wm) and since Disney uses a separable masking
  // function, we get G1(wi, wm) * G1(wo, wm) / G1(wi, wm) = G1(wo, wm) as our
  // weight.
  float G1v = GGX_G1(Wo, H, tax, tay);

  float pdf;

  float3 wi;
  if (rnd(seed) <= F) {
    Wi = normalize(-Wo + 2.f * dot(Wo, H) * H);
  } else {
    if (thin) {
      // -- When the surface is thin so it refracts into and then out of the
      // surface during this shading event.
      // -- So the ray is just reflected then flipped and we use the sqrt of the
      // surface color.
      Wi = -Wo + 2.f * dot(Wo, H) * H;
      Wi.y = -wi.y;

    } else {
      if (Transmit(H, Wo, relativeIOR, Wi)) {
        // do nothing
      } else {
        wi = -Wo + 2.f * dot(Wo, H) * H;
      }
    }

    Wi = normalize(Wi);
  }

  if (CosTheta(wi) == 0.0f) return make_float3(0.f);

  // convert wi back to world space
  // TODO: check worldToTangent transform
  // return normalize(MatrixMultiply(wi,
  // MatrixTranspose(surface.worldToTangent)));
  return normalize(Wi);
}

RT_FUNCTION float3 Evaluate_Transmission(const Disney_Parameters& surface,
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

  float d = GGX_D(H, ax, ay);
  float gl = GGX_G1(Wi, H, ax, ay);
  float gv = GGX_G1(Wo, H, ax, ay);

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

RT_FUNCTION float Evaluate_RetroDiffuse(const Disney_Parameters& surface,
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

RT_FUNCTION float Evaluate_Diffuse(const Disney_Parameters& surface,
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
  float retro = Evaluate_RetroDiffuse(surface, Wo, H, Wi);
  float subsurface = lerp(lambert, hKrueger, thin ? surface.flatness : 0.f);

  return INV_PI * (retro + subsurface * (1.f - 0.5f * fl) * (1.f - 0.5f * fv));
}

#endif