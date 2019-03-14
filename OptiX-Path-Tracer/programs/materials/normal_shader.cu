#include "material.cuh"

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(Ray, ray, rtCurrentRay, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );

/*! the attributes we use to communicate between intersection programs and hit
 * program */
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );

rtDeclareVariable(int, useShadingNormal, , );

RT_PROGRAM void closest_hit() {
  // assign material params to prd
  prd.matType = Normal_Material;

  // check if we should use geometric or shading normals
  if (useShadingNormal) {
    prd.geometric_normal = hit_rec.geometric_normal;
    prd.shading_normal = hit_rec.shading_normal;
  } else
    prd.geometric_normal = prd.shading_normal = hit_rec.geometric_normal;

  // set color based on normal value
  prd.attenuation = prd.shading_normal * 0.5f + make_float3(0.5f);
}

RT_CALLABLE_PROGRAM float3 BRDF_Sample(PDFParams &pdf, uint &seed) {
  return make_float3(0.f);
}

RT_CALLABLE_PROGRAM float BRDF_PDF(PDFParams &pdf) { return 1.f; }

RT_CALLABLE_PROGRAM float3 BRDF_Evaluate(PDFParams &pdf) {
  return make_float3(1.f);
}
