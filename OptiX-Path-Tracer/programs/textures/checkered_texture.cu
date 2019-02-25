#include "texture.cuh"

rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, odd, , );
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, even, , );

RT_CALLABLE_PROGRAM float3 sample_texture(float u, float v, float3 p) {
  float sines = sin(10 * p.x) * sin(10 - p.y) * sin(10 * p.z);

  if (sines < 0)
    return odd(u, v, p);
  else
    return even(u, v, p);
}