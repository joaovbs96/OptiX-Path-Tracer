#include "pdf.h"

rtDeclareVariable(rtCallableProgramId<float(PDFParams &)>, p0_value, , );
rtDeclareVariable(rtCallableProgramId<float(PDFParams &)>, p1_value, , );

RT_CALLABLE_PROGRAM float mixture_value(PDFParams &in) {
  return 0.5f * p0_value(in) + 0.5f * p1_value(in);
}

rtDeclareVariable(rtCallableProgramId<float3(PDFParams &, XorShift32 &)>,
                  p0_generate, , );
rtDeclareVariable(rtCallableProgramId<float3(PDFParams &, XorShift32 &)>,
                  p1_generate, , );

RT_CALLABLE_PROGRAM float3 mixture_generate(PDFParams &in, XorShift32 &rnd) {
  if (rnd() < 0.5f)
    return p0_generate(in, rnd);
  else
    return p1_generate(in, rnd);
}