#include "pdfs/pdf.cuh"
#include "prd.cuh"
#include "random.cuh"

// https://github.com/knightcrawler25/Optix-PathTracer
// https://computergraphics.stackexchange.com/questions/4979/what-is-importance-sampling
// https://computergraphics.stackexchange.com/questions/5152/progressive-path-tracing-with-explicit-light-sampling

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(Ray, ray, rtCurrentRay, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData_Shadow, prd_shadow, rtPayload, );
rtDeclareVariable(rtObject, world, , );

rtDeclareVariable(int, is_light, , );

RT_PROGRAM void any_hit() {
  if (is_light)
    prd_shadow.inShadow = false;
  else
    prd_shadow.inShadow = true;
  rtTerminateRay();
}