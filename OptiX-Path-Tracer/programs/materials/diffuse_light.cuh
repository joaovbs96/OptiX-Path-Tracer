#include "material.cuh"

struct Diffuse_Light_Parameters {
  float3 color;
};

RT_FUNCTION float3 Sample(const Diffuse_Light_Parameters &surface,
                          const float3 &P,   // next ray origin
                          const float3 &Wo,  // prev ray direction
                          const float3 &N,   // shading normal
                          uint &seed) {
  return random_on_unit_sphere(seed);
}

RT_FUNCTION float3 Evaluate(const Diffuse_Light_Parameters &surface,
                            const float3 &P,   // next ray origin
                            const float3 &Wo,  // prev ray direction
                            const float3 &Wi,  // next ray direction
                            const float3 &N,
                            float &pdf) {  // shading normal
  pdf = 1.f;
  return surface.color;
}