#include "material.h"

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(rtObject, world, , );

/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(float3, hit_rec_normal, attribute hit_rec_normal, );
rtDeclareVariable(float3, hit_rec_p, attribute hit_rec_p, );
rtDeclareVariable(float, hit_rec_u, attribute hit_rec_u, );
rtDeclareVariable(float, hit_rec_v, attribute hit_rec_v, );

/*! and finally - that particular material's parameters */
rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, sample_texture, , );


/*! the actual scatter function - in Pete's reference code, that's a
  virtual function, but since we have a different function per program
  we do not need this here */
  inline __device__ bool scatter(const optix::Ray &ray_in,
                                 DRand48 &rndState,
                                 vec3f &scattered_origin,
                                 vec3f &scattered_direction,
                                 vec3f &attenuation,
                                 float &pdf) {
  return false;
}

inline __device__ float scattering_pdf(){
  return false;
}

inline __device__ float3 emitted(){
  if(dot(hit_rec_normal, ray.direction) < 0.f)
    return sample_texture(hit_rec_u, hit_rec_v, hit_rec_p);
  else
    return make_float3(0.f);
}

RT_PROGRAM void closest_hit() {
  prd.out.emitted = emitted();
  prd.out.normal = hit_rec_normal;
  prd.out.scatterEvent
    = scatter(ray,
              *prd.in.randState,
              prd.out.scattered_origin,
              prd.out.scattered_direction,
              prd.out.attenuation,
              prd.out.pdf)
    ? rayGotBounced
    : rayGotCancelled;
  prd.out.scattered_pdf = scattering_pdf();
}
