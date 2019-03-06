#include "texture.cuh"

rtDeclareVariable(int, size, , );
rtBuffer<rtCallableProgramId<float3(float, float, float3, int)> >
    texture_vector;

RT_CALLABLE_PROGRAM float3 sample_texture(float u, float v, float3 p, int i) {
  if (i >= size || i < 0)
    return make_float3(0.f);
  else
    return texture_vector[i](u, v, p, 0);
}