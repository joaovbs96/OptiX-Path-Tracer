#pragma once

#include "../math/trigonometric.cuh"
#include "../pdfs/pdf.cuh"

// Beckmann Microfacet Distribution functions from PBRT
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/microfacet.cpp
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/microfacet.h

RT_FUNCTION float Beckmann_Roughness(float roughness) {
  roughness = max(roughness, 0.001f);
  float x = logf(roughness);

  return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
         0.000640711f * x * x * x * x;
}

RT_FUNCTION float3 Beckmann_Sample(float3 origin, float2 random, float nu, float nv) {
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
    phi = atanf(nv / nu * tanf(2.f * PI_F * random.y + 0.5f * PI_F));
    if (random.y > 0.5f) phi += PI_F;
    
    float sinPhi = sinf(phi), cosPhi = cosf(phi);
    
    tan2Theta = -logSample;
    tan2Theta /= (cosPhi * cosPhi / (nu * nu) + sinPhi * sinPhi / (nv * nv));
  }

  // Map sampled Beckmann angles to normal direction _wh_
  float cosTheta = 1.f / sqrtf(1.f + tan2Theta);
  float sinTheta = sqrtf(max(0.f, 1.f - cosTheta * cosTheta));
  float3 H = Spherical_Vector(sinTheta, cosTheta, phi);
  if (!SameHemisphere(origin, H)) H = -H;

  return H;
}

RT_FUNCTION float Beckmann_D(const float3& H, float nu, float nv) {
  float tan2Theta = Tan2Theta(H);
  if (isinf(tan2Theta)) return 0.f;

  float cos2Theta = Cos2Theta(H);
  float expo = -tan2Theta * (Cos2Phi(H) / (nu * nu) + Sin2Phi(H) / (nv * nv));

  return  expf(expo) / (PI_F * nu * nv * cos2Theta * cos2Theta);
}

RT_FUNCTION float Beckmann_PDF(const float3& H, float nu, float nv) {
  return Beckmann_D(H, nu, nv) * AbsCosTheta(H);
}