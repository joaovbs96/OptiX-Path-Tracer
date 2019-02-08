#include "material.h"

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );

/*! the attributes we use to communicate between intersection programs and hit
 * program */
rtDeclareVariable(Hit_Record, hit_rec, attribute hit_rec, );

/*! and finally - that particular material's parameters */
rtBuffer<rtCallableProgramId<float3(float, float, float3)> > sample_texture;

inline __device__ bool scatter(const optix::Ray &ray_in) {
  prd.out.is_specular = true;  // TODO: It's not specular, but shouldn't it be
                               // treated in the same way?
  prd.out.origin = hit_rec.p;
  prd.out.direction = random_in_unit_sphere(*prd.in.randState);
  prd.out.normal = hit_rec.normal;
  prd.out.attenuation =
      sample_texture[hit_rec.index](hit_rec.u, hit_rec.v, hit_rec.p);
  prd.out.type = Isotropic;

  return true;
}

inline __device__ float3 emitted() { return make_float3(0.f, 0.f, 0.f); }

RT_PROGRAM void closest_hit() {
  prd.out.emitted = emitted();
  prd.out.scatterEvent = scatter(ray) ? rayGotBounced : rayGotCancelled;
}
