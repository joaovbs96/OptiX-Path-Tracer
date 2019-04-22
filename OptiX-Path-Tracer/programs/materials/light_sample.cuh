#ifndef LIGHTSAMPLECUH
#define LIGHTSAMPLECUH

#include "ashikhmin_shirley.cuh"
#include "diffuse_light.cuh"
#include "isotropic.cuh"
#include "lambertian.cuh"
#include "material.cuh"
#include "oren_nayar.cuh"
#include "torrance_sparrow.cuh"

// Light sampling callable programs
rtDeclareVariable(int, numLights, , );
rtBuffer<float3> Light_Emissions;
rtBuffer<rtCallableProgramId<float3(const float3 &,  // P
                                    const float3 &,  // Wo
                                    const float3 &,  // N
                                    uint &)>>        // rnd seed
    Light_Sample;

rtBuffer<rtCallableProgramId<float(const float3 &,    // P
                                   const float3 &,    // Wo
                                   const float3 &,    // Wi
                                   const float3 &)>>  // N
    Light_PDF;

RT_FUNCTION float PowerHeuristic(unsigned int numf, float fPdf,
                                 unsigned int numg, float gPdf) {
  float f = numf * fPdf;
  float g = numg * gPdf;

  return (f * f) / (f * f + g * g);
}

// TODO: direct light not working
template <typename T>
RT_FUNCTION float3 Direct_Light(T &surface,        // surface parameters
                                const float3 &P,   // next ray origin
                                const float3 &Wo,  // previous ray direction
                                const float3 &N,   // surface normal
                                bool isLight, uint &seed) {
  float3 directLight = make_float3(0.f);

  // return black if there's no light
  if (numLights == 0) return make_float3(0.f);

  // ramdomly pick one light and multiply the result by the number of lights
  // it's the same as dividing by the PDF if they have the same probability
  int index = ((int)(rnd(seed) * numLights)) % numLights;

  // return black if there's just one light and we just hit it
  if (isLight && numLights == 1) return make_float3(0.f);

  // Sample Light
  float3 emission = Light_Emissions[index];
  float3 Wi = Light_Sample[index](P, Wo, N, seed);
  float lightPDF = Light_PDF[index](P, Wo, Wi, N);

  // only sample if surface normal is in the light direction
  if (dot(Wi, N) < 0.f) return make_float3(0.f);

  // Check if light is occluded
  PerRayData_Shadow prdShadow;
  Ray shadowRay = make_Ray(/* origin   : */ P,
                           /* direction: */ Wi,
                           /* ray type : */ 1,
                           /* tmin     : */ 1e-3f,
                           /* tmax     : */ RT_DEFAULT_MAX);
  rtTrace(world, shadowRay, prdShadow);

  // if light is occluded, return black
  if (prdShadow.inShadow) return make_float3(0.f);

  // Multiple Importance Sample

  // Sample light
  if (lightPDF != 0.f && !isNull(emission)) {
    float matPDF;
    float3 matValue = Evaluate(surface, P, Wo, Wi, N, matPDF);

    if (matPDF != 0.f && !isNull(matValue)) {
      float weight = PowerHeuristic(1, lightPDF, 1, matPDF);
      directLight += matValue * emission * weight / lightPDF;
    }
  }

  // Sample BRDF
  Wi = Sample(surface, P, Wo, N, seed);
  float matPDF;
  float3 matValue = Evaluate(surface, P, Wo, Wi, N, matPDF);

  if (matPDF != 0.f && !isNull(matValue)) {
    lightPDF = Light_PDF[index](P, Wo, Wi, N);

    // we didn't hit anything, ignore BRDF sample
    if (!lightPDF || isNull(emission)) return directLight;

    float weight = PowerHeuristic(1, matPDF, 1, lightPDF);
    directLight += matValue * emission * weight / matPDF;
  }

  return directLight;
}

#endif