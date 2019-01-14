#include "material.h"

// the implicit state's ray we will intersect against
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

// the per ray data we operate on
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );

// the attributes we use to communicate between intersection programs and hit program
rtDeclareVariable(Hit_Record, hit_rec, attribute hit_rec, );

// and finally - that particular material's parameters
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, sample_texture, , );


inline __device__ bool scatter(const optix::Ray &ray_in) {
  return false;
}

RT_CALLABLE_PROGRAM float scattering_pdf(pdf_in &in){
  return false;
}

inline __device__ float3 emitted(){
  if(dot(hit_rec.normal, ray.direction) < 0.f)
    return sample_texture(hit_rec.u, hit_rec.v, hit_rec.p.as_float3());
  else
    return make_float3(0.f);
}

RT_PROGRAM void closest_hit() {
  prd.out.type = Diffuse_Light;
  prd.out.emitted = emitted();
  prd.out.normal = hit_rec.normal;
  prd.out.scatterEvent = scatter(ray) ? rayGotBounced : rayGotCancelled;
}
