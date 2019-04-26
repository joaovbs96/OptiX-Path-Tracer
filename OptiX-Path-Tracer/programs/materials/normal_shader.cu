#include "material.cuh"

///////////////////////////
// --- Normal Shader --- //
///////////////////////////

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                 // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );             // ray PRD
rtDeclareVariable(rtObject, world, , );                      // scene graph
rtDeclareVariable(HitRecord, hit_rec, attribute hit_rec, );  // from geometry

// Material Parameters
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