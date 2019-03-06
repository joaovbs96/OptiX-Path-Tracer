#include "texture.cuh"

rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>, odd,
                  , );
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>, even,
                  , );

RT_CALLABLE_PROGRAM float3 sample_texture(float u, float v, float3 p, int i) {
  float sines = sin(10 * p.x) * sin(10 - p.y) * sin(10 * p.z);

  if (sines < 0)
    return odd(u, v, p, 0);
  else
    return even(u, v, p, 0);
}