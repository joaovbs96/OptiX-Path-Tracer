
#include "material.cuh"

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(Ray, ray, rtCurrentRay, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );

/*! the attributes we use to communicate between intersection programs and hit
 * program */
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

/*! and finally - that particular material's parameters */
rtBuffer<rtCallableProgramId<float3(float, float, float3)> > sample_texture;
rtDeclareVariable(float, nv, , );
rtDeclareVariable(float, nu, , );
// TODO: change the index of the geometry to the index of its first texture of
// the buffer. In a similar manner, make the buffer available to the whole
// context, not just the material, so other materials may be able to use
// the buffer as well.

// Paper & Tech Report
// https://www.cs.utah.edu/~shirley/papers/jgtbrdf.pdf
// https://www.cs.utah.edu/docs/techreports/2000/pdf/UUCS-00-014.pdf

// implementation for reference:
// https://github.com/JerryCao1985/SORT/blob/master/src/bsdf/ashikhmanshirley.cpp
// https://github.com/JerryCao1985/SORT/blob/master/src/bsdf/ashikhmanshirley.h

// https://github.com/JerryCao1985/SORT/blob/547286d552dc0e555d8144fd28bfae58535ad0d8/src/bsdf/microfacet.cpp
// https://github.com/JerryCao1985/SORT/blob/547286d552dc0e555d8144fd28bfae58535ad0d8/src/bsdf/microfacet.h

RT_PROGRAM void closest_hit() {
  prd.matType = Anisotropic_Phong;
  prd.isSpecular = false;  // TODO: ?
  prd.scatterEvent = rayGotBounced;

  prd.origin = hit_rec.p;
  prd.normal = hit_rec.normal;

  int index = hit_rec.index;
  prd.emitted = make_float3(0.f);
  prd.attenuation = sample_texture[index](hit_rec.u, hit_rec.v, hit_rec.p);
}

RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  float3 temp;

  cosine_sample_hemisphere(rnd(seed), rnd(seed), temp);

  Onb uvw(pdf.normal);
  uvw.inverse_transform(temp);

  pdf.direction = temp;

  return pdf.direction;
}

RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) {
  if (!SameHemiSphere(wo, wi)) return 0.f;
  // if (!doubleSided && !PointingUp(wo)) return 0.f;

  const auto wh = Normalize(wi + wo);
  const auto pdf_wh = distribution.Pdf(wh);
  return lerp(CosHemispherePdf(wi), pdf_wh / (4.0f * Dot(wo, wh)), 0.5f);
}

RT_CALLABLE_PROGRAM float BRDF_Evaluate(PDFParams &pdf) {
  // origin -> k1
  // direction -> k2
  // normal -> n
  // Rd -> diffuse color(of the 'substrate' under the specular coating)
  // Rs -> specular color
  // nu, nv -> phong parameters

  // TODO: add a struct with 'other params' to the function call
  // TODO: change the evaluate return to float3
  float3 diffuse_color = make_float3(0.65f, 0.05f, 0.05f);  // red
  float3 specular_color = make_float3(0.2f, 0.4f, 0.9f);    // blue

  float nk1 = dot(unit_vector(pdf.normal), unit_vector(pdf.origin));
  float nk2 = dot(unit_vector(pdf.normal), unit_vector(pdf.direction));

  // diffuse component
  float3 diffuse_component = 28.f * diffuse_color;
  diffuse_component /= 23.f * PI_F;
  diffuse_component *= (make_float3(1.f) - specular_color);

  float cosine = nk1 / 2.f;
  float exponential = powf(1.f - cosine, 5.f);
  diffuse_component *= (1.f - exponential);

  cosine = nk2 / 2.f;
  exponential = powf(1.f - cosine, 5.f);
  diffuse_component *= (1.f - exponential);

  // TODO: confirm the half vector
  float3 half_vector = unit_vector(pdf.origin + pdf.direction);
  cosine = dot(half_vector, pdf.direction);
  float3 u_vector = make_float3(0.f);  // TODO: ???
  float3 v_vector = make_float3(0.f);  // TODO: ???

  // specular component
  float3 specular_component = schlick(half_vector, cosine);

  specular_component *= sqrt((nu + 1.f) * (nv + 1.f));
  specular_component /= 8.f * PI_F;

  specular_component *= dot(unit_vector(pdf.normal), half_vector);

  exponential = nu * powf(dot(half_vector, u_vector), 2.f);
  exponential += nv * powf(dot(half_vector, v_vector), 2.f);
  exponential /= 1.f - powf(dot(half_vector, pdf.normal), 2.f);
  specular_component *= exponential;

  specular_component /= cosine * ffmax(nk1, nk2);

  // fresnel
}