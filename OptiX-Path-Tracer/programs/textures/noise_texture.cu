#include "texture.cuh"

rtDeclareVariable(float, scale, , );
rtDeclareVariable(int, ax, , );

// buffer definitions
rtBuffer<float3, 1> ranvec;
rtBuffer<int, 1> perm_x;
rtBuffer<int, 1> perm_y;
rtBuffer<int, 1> perm_z;

RT_FUNCTION float perlin_interp(float3 c[2][2][2], float u, float v, float w) {
  float uu = u * u * (3 - 2 * u);
  float vv = v * v * (3 - 2 * v);
  float ww = w * w * (3 - 2 * w);
  float accum = 0;

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++) {
        float3 weight_v = make_float3(u - i, v - j, w - k);
        accum += (i * uu + (1 - i) * (1 - uu)) * (j * vv + (1 - j) * (1 - vv)) *
                 (k * ww + (1 - k) * (1 - ww)) * dot(c[i][j][k], weight_v);
      }

  return accum;
}

RT_FUNCTION float noise(float3 p) {
  float u = p.x - floor(p.x);
  float v = p.y - floor(p.y);
  float w = p.z - floor(p.z);

  int i = floor(p.x);
  int j = floor(p.y);
  int k = floor(p.z);
  float3 c[2][2][2];

  for (int di = 0; di < 2; di++)
    for (int dj = 0; dj < 2; dj++)
      for (int dk = 0; dk < 2; dk++)
        c[di][dj][dk] = ranvec[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^
                               perm_z[(k + dk) & 255]];

  return perlin_interp(c, u, v, w);
}

RT_FUNCTION float turb(float3 p) {
  float accum = 0;
  float3 temp_p = p;
  float weight = 1.0;

  for (int i = 0; i < 7; i++) {
    accum += weight * noise(temp_p);
    weight *= 0.5;
    temp_p *= 2;
  }

  return fabs(accum);
}

RT_CALLABLE_PROGRAM float3 sample_texture(float u, float v, float3 p, int i) {
  float sinValue;

  // get value of sin term according to chosen axis
  switch (AXIS(ax)) {
    case X_AXIS:
      sinValue = sin(scale * p.x + 5 * turb(scale * p));
    case Y_AXIS:
      sinValue = sin(scale * p.y + 5 * turb(scale * p));
    case Z_AXIS:
      sinValue = sin(scale * p.z + 5 * turb(scale * p));
  }

  return make_float3(1.f) * 0.5 * (1 + sinValue);
}