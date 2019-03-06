#include "texture.cuh"

rtTextureSampler<float4, 2> data;

RT_CALLABLE_PROGRAM float3 sample_texture(float u, float v, float3 p, int i) {
  return make_float3(tex2D(data, u, v));
}