#include "pdf.h"

RT_CALLABLE_PROGRAM float3 cosine_generate(pdf_in &in, XorShift32 &rnd) {
    in.uvw.build_from_w(in.normal);
    
    float3 temp = random_cosine_direction(rnd);
    in.scattered_direction = in.uvw.local(temp);
    
    return in.scattered_direction;
}

RT_CALLABLE_PROGRAM float cosine_value(pdf_in &in) {
    float cosine = dot(unit_vector(in.scattered_direction), in.uvw.w);
    
    if(cosine > 0.f)
        return cosine / PI_F;
    else
        return 0.f;
}