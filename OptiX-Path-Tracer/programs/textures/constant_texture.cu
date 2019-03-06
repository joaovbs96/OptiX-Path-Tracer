#include "texture.cuh"

rtDeclareVariable(float3, color, , );

RT_CALLABLE_PROGRAM float3 sample_texture(float u, float v, float3 p, int i) {
  return color;
}