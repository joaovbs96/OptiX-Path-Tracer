#include "texture.cuh"

rtDeclareVariable(float3, colorA, , );
rtDeclareVariable(float3, colorB, , );
rtDeclareVariable(float3, colorC, , );

RT_CALLABLE_PROGRAM float3 sample_texture(float u, float v, float3 p, int i) {
  const float3 unit_direction = normalize(p);
  const float x = fabsf(unit_direction.x);
  const float y = fabsf(unit_direction.y);
  const float z = fabsf(unit_direction.z);

  return x * colorA + y * colorB + z * colorC;
}
