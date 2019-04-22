#include "material.cuh"

struct Oren_Nayar_Parameters {
  float3 color;
  float rA, rB;
};

RT_FUNCTION float3 Sample(const Oren_Nayar_Parameters &surface,
                          const float3 &P,   // next ray origin
                          const float3 &Wo,  // prev ray direction
                          const float3 &N,   // shading normal
                          uint &seed) {
  float3 Wi;
  cosine_sample_hemisphere(rnd(seed), rnd(seed), Wi);

  Onb uvw(N);
  uvw.inverse_transform(Wi);

  return Wi;
}

RT_FUNCTION float PDF(const Oren_Nayar_Parameters &surface,
                      const float3 &P,    // next ray origin
                      const float3 &Wo,   // prev ray direction
                      const float3 &Wi,   // next ray direction
                      const float3 &N) {  // shading normal
  return dot(normalize(Wi), normalize(N)) / PI_F;
}

RT_FUNCTION float3 Evaluate(const Oren_Nayar_Parameters &surface,
                            const float3 &P,   // next ray origin
                            const float3 &Wo,  // prev ray direction
                            const float3 &Wi,  // next ray direction
                            const float3 &N,
                            float &pdf) {  // shading normal
  float3 WiN = normalize(Wi);

  float sinThetaI = SinTheta(WiN);
  float sinThetaO = SinTheta(Wo);
  // Compute cosine term of Oren-Nayar model
  float maxCos = 0;
  if (sinThetaI > 1e-4 && sinThetaO > 1e-4) {
    float sinPhiI = SinPhi(WiN), cosPhiI = CosPhi(WiN);
    float sinPhiO = SinPhi(Wo), cosPhiO = CosPhi(Wo);
    float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
    maxCos = fmaxf(0.f, dCos);
  }

  // Compute sine and tangent terms of Oren-Nayar model
  float sinAlpha, tanBeta;
  if (AbsCosTheta(WiN) > AbsCosTheta(Wo)) {
    sinAlpha = sinThetaO;
    tanBeta = sinThetaI / AbsCosTheta(WiN);
  } else {
    sinAlpha = sinThetaI;
    tanBeta = sinThetaO / AbsCosTheta(Wo);
  }

  pdf = PDF(surface, P, Wo, Wi, N);

  float rA = surface.rA;
  float rB = surface.rB;
  float3 color = surface.color;

  return color * (1.f / PI_F) * (rA + rB * maxCos * sinAlpha * tanBeta);
}