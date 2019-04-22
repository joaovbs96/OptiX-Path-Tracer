#include "material.cuh"

struct Isotropic_Parameters {
  float3 color;
};

RT_FUNCTION float3 Sample(const Isotropic_Parameters &surface,
                          const float3 &P,   // next ray origin
                          const float3 &Wo,  // prev ray direction
                          const float3 &N,   // shading normal
                          uint &seed) {
  return random_on_unit_sphere(seed);
}

RT_FUNCTION float PDF(const Isotropic_Parameters &surface,
                      const float3 &P,    // next ray origin
                      const float3 &Wo,   // prev ray direction
                      const float3 &Wi,   // next ray direction
                      const float3 &N) {  // shading normal
  return 0.25 * PI_F;
}

RT_FUNCTION float3 Evaluate(const Isotropic_Parameters &surface,
                            const float3 &P,   // next ray origin
                            const float3 &Wo,  // prev ray direction
                            const float3 &Wi,  // next ray direction
                            const float3 &N,
                            float &pdf) {  // shading normal
  pdf = PDF(surface, P, Wo, Wi, N);
  return 0.25 * PI_F * surface.color;
}