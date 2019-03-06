#include "pdf.cuh"

rtDeclareVariable(int, size, , );

rtBuffer<rtCallableProgramId<float(PDFParams&)>> values;

RT_CALLABLE_PROGRAM float value(PDFParams& in) {
  float sum = 0.f;

  for (int i = 0; i < size; i++) sum += values[i](in);
  sum /= size;

  return sum;
}

rtBuffer<rtCallableProgramId<float3(PDFParams&, uint&)>> generators;

RT_CALLABLE_PROGRAM float3 generate(PDFParams& in, uint& seed) {
  int index = int(rnd(seed) * size);
  return generators[index](in, seed);
}