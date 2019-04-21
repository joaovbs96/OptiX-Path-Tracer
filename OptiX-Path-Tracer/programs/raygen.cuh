#ifndef RAYGENCUH
#define RAYGENCUH

#include "prd.cuh"
#include "sampling.cuh"
#include "vec.hpp"


// Light sampling callable programs
rtDeclareVariable(int, numLights, , );
rtBuffer<float3> Light_Emissions;
rtBuffer<rtCallableProgramId<float3(const float3&,  // P
                                    const float3&,  // Wo
                                    const float3&,  // N
                                    uint&)>>        // rnd seed
    Light_Sample;
rtBuffer<rtCallableProgramId<float(const float3&,    // P
                                   const float3&,    // Wo
                                   const float3&,    // Wi
                                   const float3&)>>  // N
    Light_PDF;

// BRDF sampling callable programs
rtBuffer<rtCallableProgramId<float3(const BRDFParameters&,  // surface params
                                    const float3&,          // P
                                    const float3&,          // Wo
                                    const float3&,          // N
                                    uint&)>>                // rnd seed
    BRDF_Sample;
rtBuffer<rtCallableProgramId<float(const BRDFParameters&,  // surface params
                                   const float3&,          // P
                                   const float3&,          // Wo
                                   const float3&,          // Wi
                                   const float3&)>>        // N
    BRDF_PDF;
rtBuffer<rtCallableProgramId<float3(const BRDFParameters&,  // surface params
                                    const float3&,          // P
                                    const float3&,          // Wo
                                    const float3&,          // Wi
                                    const float3&)>>        // N
    BRDF_Evaluate;

#endif