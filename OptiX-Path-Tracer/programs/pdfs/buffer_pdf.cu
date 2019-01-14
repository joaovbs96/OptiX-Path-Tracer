#include "pdf.h"

rtDeclareVariable(int, size, , );

rtBuffer< rtCallableProgramId<float(pdf_in&)> > values;

RT_CALLABLE_PROGRAM float buffer_value(pdf_in &in) {
    float sum = 0.f;
    
    for(int i = 0; i < size; i++)
        sum += values[i](in);
    sum /= size;

    return sum;
}

rtBuffer< rtCallableProgramId<float3(pdf_in&, DRand48&)> > generators;

RT_CALLABLE_PROGRAM float3 buffer_generate(pdf_in &in, DRand48 &rnd) {
    int index = int(rnd() * size);
    return generators[index](in, rnd);
}