#pragma once

#include "../math/trigonometric.cuh"
#include "../pdfs/pdf.cuh"

// Beckmann Microfacet Distribution functions from PBRT
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/microfacet.cpp
// https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/microfacet.h

RT_FUNCTION float Beckmann_Roughness(float roughness) {
  roughness = max(roughness, 0.001f);
  float x = log(roughness);

  return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
         0.000640711f * x * x * x * x;
}

// TODO: adapt
RT_FUNCTION float3 Beckmann_Sample(float3 origin, float u, float v) {
  /*// Sample full distribution of normals for Beckmann distribution

  // Compute $\tan^2 \theta$ and $\phi$ for Beckmann distribution sample
  Float tan2Theta, phi;
  if (alphax == alphay) {
    Float logSample = std::log(1 - u[0]);
    DCHECK(!std::isinf(logSample));
    tan2Theta = -alphax * alphax * logSample;
    phi = u[1] * 2 * Pi;
  } else {
    // Compute _tan2Theta_ and _phi_ for anisotropic Beckmann
    // distribution
    Float logSample = std::log(1 - u[0]);
    DCHECK(!std::isinf(logSample));
    phi = std::atan(alphay / alphax * std::tan(2 * Pi * u[1] + 0.5f * Pi));
    if (u[1] > 0.5f) phi += Pi;
    Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
    Float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
    tan2Theta =
        -logSample / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
  }

  // Map sampled Beckmann angles to normal direction _wh_
  Float cosTheta = 1 / std::sqrt(1 + tan2Theta);
  Float sinTheta = std::sqrt(std::max((Float)0, 1 - cosTheta * cosTheta));
  Vector3f wh = SphericalDirection(sinTheta, cosTheta, phi);
  if (!SameHemisphere(wo, wh)) wh = -wh;
  return wh;*/
}

RT_FUNCTION float Beckmann_D(const float3& wh, float nu, float nv) {
  float tan2Theta = Tan2Theta(wh);
  if (isinf(tan2Theta)) return 0.f;

  float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

  return expf(-tan2Theta *
              (Cos2Phi(wh) / (nu * nu) + Sin2Phi(wh) / (nv * nv))) /
         (PI_F * nu * nv * cos4Theta);
}

RT_FUNCTION float Beckmann_PDF(const float3& wh, float nu, float nv) const {
  /*const float3& wo
  if (sampleVisibleArea)
    return D(wh) * G1(wo) * AbsDot(wo, wh) / AbsCosTheta(wo);
  else*/
  return Beckmann_D(wh, nu, nv) * AbsCosTheta(wh);
}