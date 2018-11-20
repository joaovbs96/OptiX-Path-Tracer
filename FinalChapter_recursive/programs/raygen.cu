// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// optix code:
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "prd.h"
#include "sampling.h"

/*! the 'builtin' launch index we need to render a frame */
rtDeclareVariable(uint2, pixelID,   rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim,   );

/*! the ray related state */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData, prd, rtPayload, );

/*! the 2D, float3-type color frame buffer we'll write into */
rtBuffer<float3, 2> fb;

rtDeclareVariable(int, numSamples, , );

rtDeclareVariable(rtObject, world, , );

rtDeclareVariable(float3, camera_lower_left_corner, , );
rtDeclareVariable(float3, camera_horizontal, , );
rtDeclareVariable(float3, camera_vertical, , );
rtDeclareVariable(float3, camera_origin, , );
rtDeclareVariable(float3, camera_u, , );
rtDeclareVariable(float3, camera_v, , );
rtDeclareVariable(float, camera_lens_radius, , );

struct Camera {
  static __device__ optix::Ray generateRay(float s, float t, DRand48 &rnd) 
  {
    const vec3f rd = camera_lens_radius * random_in_unit_disk(rnd);
    const vec3f lens_offset = camera_u * rd.x + camera_v * rd.y;
    const vec3f origin = camera_origin + lens_offset;
    const vec3f direction
      = camera_lower_left_corner
      + s * camera_horizontal
      + t * camera_vertical
      - origin;
    return optix::make_Ray(/* origin   : */ origin.as_float3(),
                           /* direction: */ direction.as_float3(),
                           /* ray type : */ 0,
                           /* tmin     : */ 1e-6f,
                           /* tmax     : */ RT_DEFAULT_MAX);
  }
};

inline __device__ vec3f color(optix::Ray ray, DRand48 &rnd)
{
  PerRayData prd;
  prd.randState = &rnd;
  prd.depth = 0;
  rtTrace(world, ray, prd);
  return prd.color;
}

/*! the actual ray generation program - note this has no formal
  function parameters, but gets its paramters throught the 'pixelID'
  and 'pixelBuffer' variables/buffers declared above */
RT_PROGRAM void renderPixel()
{
  int pixel_index = pixelID.y * launchDim.x + pixelID.x;
  vec3f col(0.f, 0.f, 0.f);
  DRand48 rnd;
  rnd.init(pixel_index);
  for (int s = 0; s < numSamples; s++) {
    float u = float(pixelID.x + rnd()) / float(launchDim.x);
    float v = float(pixelID.y + rnd()) / float(launchDim.y);
    optix::Ray ray = Camera::generateRay(u, v, rnd);
    col += color(ray, rnd);
  }
  col = col / float(numSamples);

  fb[pixelID] = col.as_float3();
}

