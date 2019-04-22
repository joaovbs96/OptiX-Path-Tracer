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
  // set color based on normal value
  // check if we should use geometric or shading normals
  if (useShadingNormal) {
    prd.radiance = hit_rec.shading_normal * 0.5f + make_float3(0.5f);
  } else {
    prd.radiance = hit_rec.geometric_normal * 0.5f + make_float3(0.5f);
  }

  // Assign parameters to PRD
  prd.scatterEvent = rayGotCancelled;
}