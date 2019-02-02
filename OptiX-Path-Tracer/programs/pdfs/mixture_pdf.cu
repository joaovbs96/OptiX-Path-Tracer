#include "pdf.h"

rtDeclareVariable(rtCallableProgramId<float(pdf_in&)>, p0_value, , );
rtDeclareVariable(rtCallableProgramId<float(pdf_in&)>, p1_value, , );

RT_CALLABLE_PROGRAM float mixture_value(pdf_in &in) {
    return 0.5f * p0_value(in) + 0.5f * p1_value(in);
}

rtDeclareVariable(rtCallableProgramId<float3(pdf_in&, XorShift32&)>, p0_generate, , );
rtDeclareVariable(rtCallableProgramId<float3(pdf_in&, XorShift32&)>, p1_generate, , );

RT_CALLABLE_PROGRAM float3 mixture_generate(pdf_in &in, XorShift32 &rnd) {
    if (rnd() < 0.5f)
        return p0_generate(in, rnd);
    else
        return p1_generate(in, rnd);
}