#include "pdf.cuh"

RT_CALLABLE_PROGRAM float3 cosine_generate(PDFParams &pdf, uint &seed) {
  float3 temp;

  cosine_sample_hemisphere(rnd(seed), rnd(seed), temp);

  Onb uvw(pdf.normal);
  uvw.inverse_transform(temp);

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