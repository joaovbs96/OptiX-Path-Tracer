#include "pdf.h"

rtDeclareVariable(int, size, , );

rtBuffer<rtCallableProgramId<float(PDFParams&)>> values;

RT_CALLABLE_PROGRAM float buffer_value(PDFParams& in) {
  float sum = 0.f;

  for (int i = 0; i < size; i++) sum += values[i](in);
  sum /= size;

  return sum;
}

rtBuffer<rtCallableProgramId<float3(PDFParams&, XorShift32&)>> generators;

RT_CALLABLE_PROGRAM float3 buffer_generate(PDFParams& in, XorShift32& rnd) {
  int index = int(rnd() * size);
  return generators[index](in, rnd);
}