#include "material.cuh"

// the implicit state's ray we will intersect against
rtDeclareVariable(Ray, ray, rtCurrentRay, );

// the per ray data we operate on
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );

// the attributes we use to communicate between intersection programs and hit
// program
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

rtDeclareVariable(rtCallableProgramId<float3(float, float, float3, int)>,
                  sample_texture, , );

RT_PROGRAM void closest_hit() {
  // get material params from buffer
  int texIndex = hit_rec.index;

  // assign material params to prd
  prd.matType = Diffuse_Light_Material;
  prd.isSpecular = false;
  prd.scatterEvent = rayGotCancelled;

  // assign hit params to prd
  prd.normal = hit_rec.normal;

  // assign emission prd
  if (dot(prd.normal, ray.direction) < 0.f)
    prd.emitted = sample_texture(hit_rec.u, hit_rec.v, hit_rec.p, texIndex);
  else
    prd.emitted = make_float3(0.f);
}

RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  return make_float3(1.f);
}

RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) { return 1.f; }

RT_CALLABLE_PROGRAM float BRDF_Evaluate(PDFParams &pdf) { return 1.f; }
