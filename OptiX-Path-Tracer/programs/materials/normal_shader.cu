#include "material.cuh"

///////////////////////////
// --- Normal Shader --- //
///////////////////////////

// OptiX Context objects
rtDeclareVariable(Ray, ray, rtCurrentRay, );                // current ray
rtDeclareVariable(PerRayData, prd, rtPayload, );            // ray PRD
rtDeclareVariable(rtObject, world, , );                     // scene graph
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );  // hit distance

// Intersected Geometry Parameters
rtDeclareVariable(HitRecord_Function, Get_HitRecord, , );  // HitRecord function
rtDeclareVariable(int, geo_index, attribute geo_index, );  // primitive index
rtDeclareVariable(float2, bc, attribute bc, );  // triangle barycentrics

// Material Parameters
rtDeclareVariable(int, useShadingNormal, , );

// Assigns material and hit parameters to PRD
RT_PROGRAM void closest_hit() {
  HitRecord rec = Get_HitRecord(geo_index, ray, t_hit, bc);

  // set color based on normal value
  // check if we should use geometric or shading normals
  if (useShadingNormal) {
    prd.radiance = rec.shading_normal * 0.5f + make_float3(0.5f);
  } else {
    prd.radiance = rec.geometric_normal * 0.5f + make_float3(0.5f);
  }

  // Assign parameters to PRD
  prd.scatterEvent = rayGotCancelled;
}