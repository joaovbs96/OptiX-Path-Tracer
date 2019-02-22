#include "material.h"

// the implicit state's ray we will intersect against
rtDeclareVariable(Ray, ray, rtCurrentRay, );

// the per ray data we operate on
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );

// the attributes we use to communicate between intersection programs and hit
// program
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

// and finally - that particular material's parameters
rtBuffer<rtCallableProgramId<float3(float, float, float3)> > sample_texture;

RT_PROGRAM void closest_hit() {
  prd.matType = Diffuse_Light_Material;
  prd.isSpecular = false;
  prd.scatterEvent = rayGotCancelled;

  prd.normal = hit_rec.normal;

  int index = hit_rec.index;
  if (dot(prd.normal, ray.direction) < 0.f)
    prd.emitted = sample_texture[index](hit_rec.u, hit_rec.v, hit_rec.p);
  else
    prd.emitted = make_float3(0.f);
}

RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, XorShift32 &rnd) {
  return make_float3(1.f);
}

RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) { return 1.f; }

RT_CALLABLE_PROGRAM float BRDF_Evaluate(PDFParams &pdf) { return 1.f; }
