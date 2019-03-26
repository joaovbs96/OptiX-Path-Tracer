#pragma once

#include "../math/trigonometric.cuh"
#include "../pdfs/pdf.cuh"

// Sampling Ashikhmin-Shirley Quadrant - From Blender's implementation
// https://developer.blender.org/diffusion/C/browse/master/src/kernel/closure/bsdf_ashikhmin_shirley.h

RT_FUNCTION void Sample_Quadrant(float nu, float nv, float randX, float randY, float &phi, float &theta){
  phi = atanf(sqrtf((nu + 1.f) / (nv + 1.f)) * tanf(2.f * PI_F * randX));
  
  float cos_phi = cosf(phi);
  float sin_phi = sinf(phi);
  
  theta = powf(randY, 1.0f / (nu * cos_phi * cos_phi + nv * sin_phi * sin_phi + 1.f));
}

// Beckmann Microfacet Distribution functions from PBRT
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/microfacet.cpp
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/microfacet.h

RT_FUNCTION float3 Beckmann_Sample(float3 origin, float2 random, float nu,
                                   float nv) {
  // Sample full distribution of normals for Beckmann distribution

  float logSample = logf(1.f - random.x);
  if (isinf(logSample)) return make_float3(0.f);

  float tan2Theta, phi;
  if (nu == nv) {
    // Compute tan2theta and phi for Beckmann distribution sample
    tan2Theta = -nu * nu * logSample;
    phi = random.y * 2.f * PI_F;
  } else {
    // Compute tan2Theta and phi for anisotropic Beckmann distribution
    phi = atanf((nv / nu) * tanf(2.f * PI_F * random.y + 0.5f * PI_F));
    if (random.y > 0.5f) phi += PI_F;

    float sinPhi = sinf(phi), cosPhi = cosf(phi);

    tan2Theta = -logSample;
    tan2Theta /= cosPhi * cosPhi / nu * nu + sinPhi * sinPhi / nv * nv;
  }

  // Map sampled Beckmann angles to normal direction _wh_
  float cosTheta = 1.f / sqrtf(1.f + tan2Theta);
  float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
  float3 H = Spherical_Vector(sinTheta, cosTheta, phi);
  if (!Same_Hemisphere(origin, H)) H = -H;

  return H;
}

RT_FUNCTION float Beckmann_D(const float3& H, float nu, float nv) {
  float tan2Theta = Tan2Theta(H);
  if (isinf(tan2Theta)) return 0.f;

  float cos2Theta = Cos2Theta(H);
  float expo = -tan2Theta * (Cos2Phi(H) / (nu * nu) + Sin2Phi(H) / (nv * nv));

  return expf(expo) / (PI_F * nu * nv * cos2Theta * cos2Theta);
}

RT_FUNCTION float Beckmann_PDF(const float3& H, float nu, float nv) {
  return Beckmann_D(H, nu, nv) * AbsCosTheta(H);
}

// GGX Microfacet Distribution functions from JerryCao1985's SORT
// https://github.com/JerryCao1985/SORT/blob/fd2fdcbcca1c678b81047db8182e537133ffb278/src/bsdf/microfacet.cpp
// https://github.com/JerryCao1985/SORT/blob/fd2fdcbcca1c678b81047db8182e537133ffb278/src/bsdf/microfacet.h

// Other reference links:
// https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/
// https://agraphicsguy.wordpress.com/2018/07/18/sampling-anisotropic-microfacet-brdf/
// https://schuttejoe.github.io/post/ggximportancesamplingpart1/
// https://schuttejoe.github.io/post/ggximportancesamplingpart2/
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/microfacet.h
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/microfacet.cpp

// Anisotropic GGX (Trowbridge-Reitz) distribution formula(PBRT page 539)
RT_FUNCTION float GGX_D(const float3& H, float nu, float nv) {
  const float CosTheta2 = Cos2Theta(H);
  if (CosTheta2 <= 0.0f) return 0.f;

  if (nu == nv) {
    float beta = CosTheta2 * ((nu * nu) - 1) + 1;
    return (nu * nu) / (PI_F * beta * beta);
  } else {
    const float Sin2PhiNV = Sin2Phi(H) / (nv * nv);
    const float Cos2PhiNU = (1.f - Sin2Phi(H)) / (nu * nu);

    float beta = (CosTheta2 + (1.f - CosTheta2) * (Cos2PhiNU + Sin2PhiNV));
    return 1.f / (PI_F * nu * nv * beta * beta);
  }
}

// Sampling a normal respect to the NDF(PBRT 8.4.3)
RT_FUNCTION float3 GGX_Sample(float3 origin, float2 random, float nu,
                              float nv) {
  /*// stretch view
  float3 V = normalize(make_float3(nu * origin.x, origin.y, nv * origin.z));

  // orthonormal basis 
  float3 T1 = (V.y < 0.9999f) ? normalize(cross(V, make_float3(0.f, 1.f, 0.f))) : make_float3(1.f, 0.f, 0.f);
  float3 T2 = cross(T1, V);

  // sample point with polar coordinates (r, phi)
  float a = 1.f / (1.f + V.y);
  float r = sqrtf(nu);
  float phi = (nv < a) ? (nv / a) * PI_F : PI_F + (nv - a) / (1.f - a) * PI_F;
  float p1 = r * cosf(phi);
  float p2 = r * sinf(phi) * ((nv < a) ? 1.f : random.y);

  // calculate the normal in this stretched tangent space
  float3 N = p1 * T1 + p2 * T2 + sqrtf(fmaxf(0.f, 1.f - (p1 * p1) - (p2 * p2))) * V;

  // unstretch and normalize the normal
  return normalize(make_float3(nu * N.x, N.y, nv * N.z));*/


  float theta, phi;
  if (nu == nv) {
    // isotropic sampling
    phi = 2.f * PI_F * random.x;
    theta = atanf(nu * sqrtf(random.y / (1.f - random.y)));
  } else {
    // anisotropic sampling
    static const int offset[5] = {0, 1, 1, 2, 2};
    const int i = random.y == 0.25f ? 0 : (int)(random.y * 4.0f);
    phi = atanf((nv / nu) * tanf(2.f * PI_F * random.y)) + offset[i] * PI_F;

    const float sin_phi = sinf(phi);
    const float sin_phi_sq = sin_phi * sin_phi;
    const float cos_phi_sq = 1.f - sin_phi_sq;

    float beta = 1.f / (cos_phi_sq / (nu * nu) + sin_phi_sq / (nv * nv));
    theta = atanf(sqrtf(beta * random.x / (1.f - random.x)));
  }

  float3 H = Spherical_Vector(theta, phi);
  //if (!Same_Hemisphere(origin, H)) H = -H;

  return H;

  // TODO: try the approach from this article:
  // https://hal.archives-ouvertes.fr/hal-01509746/document

  /*bool flip = origin.y < 0;

  // 1. stretch the view so we are sampling as though roughness==1
  float3 stretchedOrigin = make_float3(origin.x * nu, origin.y, origin.z * nv);
  stretchedOrigin = normalize(stretchedOrigin);

  // 2. simulate P22_{wi}(slopeX, slopeY, 1, 1)
  float slopeX, slopeY;
  float cosTheta = CosTheta(stretchedOrigin);
  // TrowbridgeReitzSample11(CosTheta(stretchedOrigin), random, slopeX, slopeY);

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
    slopeY = S * z * std::sqrt(1.f + slopeX * slopeX);
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

  return H;*/
}

// Smith’s masking-shadowing function(PBRT 8.4.3)
RT_FUNCTION float GGX_G1(const float HdotV, const float alpha) {
  if (isinf(HdotV)) return 0.f;

  const float beta = alpha + (1 - alpha * alpha) * HdotV * HdotV;
  return (2.f * HdotV) / (HdotV + sqrtf(beta));
}

// also called Smith_GGX
// Visibility term of microfacet model, Smith shadow-masking function
RT_FUNCTION float GGX_G(const float3& origin, const float3& direction,
                        const float3& H, float nu, float nv) {
  return GGX_G1(dot(origin, H), nv / nu) * GGX_G1(dot(direction, H), nv / nu);
}

// PDF of sampling a specific normal direction
RT_FUNCTION float GGX_PDF(const float3& H, const float3& origin, float nu,
                          float nv) {
  return GGX_D(H, nu, nv) * AbsCosTheta(H);
  /*return GGX_D(H, nu, nv) * GGX_G1(dot(origin, H), nv / nu) *
         fabsf(dot(origin, H)) / AbsCosTheta(origin);*/
}