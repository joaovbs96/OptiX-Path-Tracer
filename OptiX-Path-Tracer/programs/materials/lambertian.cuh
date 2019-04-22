#include "material.cuh"

struct Lambertian_Parameters {
  float3 color;
};

RT_FUNCTION float3 Sample(const Lambertian_Parameters &surface,
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

RT_FUNCTION float PDF(const Lambertian_Parameters &surface,
                      const float3 &P,    // next ray origin
                      const float3 &Wo,   // prev ray direction
                      const float3 &Wi,   // next ray direction
                      const float3 &N) {  // shading normal
  return dot(normalize(Wi), normalize(N)) / PI_F;
}

RT_FUNCTION float3 Evaluate(const Lambertian_Parameters &surface,
                            const float3 &P,   // next ray origin
                            const float3 &Wo,  // prev ray direction
                            const float3 &Wi,  // next ray direction
                            const float3 &N,
                            float &pdf) {  // shading normal
  float cosine = dot(normalize(Wi), normalize(N));

  if (cosine < 0.f) {
    return make_float3(0.f);
  } else {
    pdf = PDF(surface, P, Wo, Wi, N);
    return (cosine * surface.color) / PI_F;
  }
}