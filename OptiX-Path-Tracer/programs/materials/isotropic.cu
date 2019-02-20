#include "material.h"

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

RT_PROGRAM void closest_hit() {
  prd.matType = Isotropic_Material;
  prd.isSpecular = true;  // TODO: fix sampling
  prd.scatterEvent = rayGotBounced;

  prd.origin = hit_rec.p;
  prd.normal = hit_rec.normal;
  prd.direction = random_in_unit_sphere(*prd.randState);

  int index = hit_rec.index;
  prd.emitted = make_float3(0.f);
  prd.attenuation = sample_texture[index](hit_rec.u, hit_rec.v, hit_rec.p);
}

// TODO: complete programs
RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, XorShift32 &rnd) {
  return make_float3(1.f);
}

RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) { return 1.f; }

RT_CALLABLE_PROGRAM float BRDF_Evaluate(PDFParams &pdf) { return 1.f; }
