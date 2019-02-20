#include "pdf.h"

RT_CALLABLE_PROGRAM float3 cosine_generate(PDFParams &pdf, XorShift32 &rnd) {
  Onb uvw(pdf.normal);

  float3 temp = random_cosine_direction(rnd);
  uvw.inverse_transform(temp);

  // float3 temp = unit_vector(in.normal) + random_on_unit_sphere(rnd);
  pdf.direction = temp;

  return pdf.direction;
}

RT_CALLABLE_PROGRAM float cosine_value(PDFParams &pdf) {
  float cosine = dot(unit_vector(pdf.direction), unit_vector(pdf.normal));

  if (cosine > 0.f)
    return cosine / PI_F;
  else
    return 0.f;
}