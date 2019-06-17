#include "prd.cuh"
#include "random.cuh"
#include "sampling.cuh"
#include "vec.hpp"

// https://github.com/knightcrawler25/Optix-PathTracer
// https://computergraphics.stackexchange.com/questions/4979/what-is-importance-sampling
// https://computergraphics.stackexchange.com/questions/5152/progressive-path-tracing-with-explicit-light-sampling

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_Shadow, prd_shadow, rtPayload, );
rtDeclareVariable(rtObject, world, , );

rtDeclareVariable(int, is_light, , );

RT_PROGRAM void any_hit() {
  // TODO: check if this is correct
  // only iluminate if ray is against the light normal
  if (is_light && dot(prd_shadow.normal, ray.direction) > 0.f)
    prd_shadow.inShadow = false;
  else
    prd_shadow.inShadow = true;
  rtTerminateRay();
}