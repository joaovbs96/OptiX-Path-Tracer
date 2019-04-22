
#include "material.cuh"
#include "microfacets.cuh"

struct Torrance_Sparrow_Parameters {
  float3 color;
  float nu, nv;
};

RT_FUNCTION float3 Sample(const Torrance_Sparrow_Parameters &surface,
                          const float3 &P,   // next ray origin
                          const float3 &Wo,  // prev ray direction
                          const float3 &N,   // shading normal
                          uint &seed) {
  // Get material params from input variable
  float nu = surface.nu;
  float nv = surface.nv;

  // create basis
  float3 Nn = normalize(N);
  float3 T = normalize(cross(Nn, make_float3(0.f, 1.f, 0.f)));
  float3 B = cross(T, Nn);

  // random variables
  float2 random = make_float2(rnd(seed), rnd(seed));

  // get half vector and rotate it to world space
  float3 H = normalize(GGX_Sample(Wo, random, nu, nv));
  H = H.x * B + H.y * Nn + H.z * T;

  float HdotI = dot(H, Wo);
  if (HdotI < 0.f) H = -H;

  return normalize(-Wo + 2.f * dot(Wo, H) * H);
}

RT_FUNCTION float3 Evaluate(const Torrance_Sparrow_Parameters &surface,
                            const float3 &P,   // next ray origin
                            const float3 &Wo,  // prev ray direction
                            const float3 &Wi,  // next ray direction
                            const float3 &N,   // shading normal
                            float &pdf) {
  // Get material params from input variable
  float3 Rs = surface.color;
  float nu = surface.nu;
  float nv = surface.nv;

  // create basis
  float3 Up = make_float3(0.f, 1.f, 0.f);
  float NdotI = fmaxf(dot(Up, Wi), 1e-6f), NdotO = fmaxf(dot(Up, Wo), 1e-6f);

  // half vector = (v1 + v2) / |v1 + v2|
  float3 H = Wo + Wi;
  if (isNull(H)) return make_float3(0.f);
  H = normalize(H);
  float HdotI = abs(dot(H, Wi));  // origin or direction here

  float3 F = schlick(Rs, HdotI);    // Fresnel Reflectance
  float G = GGX_G(Wo, Wi, nu, nv);  // Geometric Shadowing
  float D = GGX_D(H, nu, nv);       // Normal Distribution Function(NDF)

  pdf = GGX_PDF(H, Wo, nu, nv) / (4.f * dot(Wo, H));

  return Rs * D * G * F / (4.f * NdotI * NdotO);
}