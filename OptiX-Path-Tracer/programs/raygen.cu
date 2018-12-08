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
rtDeclareVariable(int, run, , );

rtDeclareVariable(rtObject, world, , );

rtDeclareVariable(int, light, , );

rtDeclareVariable(float3, camera_lower_left_corner, , );
rtDeclareVariable(float3, camera_horizontal, , );
rtDeclareVariable(float3, camera_vertical, , );
rtDeclareVariable(float3, camera_origin, , );
rtDeclareVariable(float3, camera_u, , );
rtDeclareVariable(float3, camera_v, , );
rtDeclareVariable(float, camera_lens_radius, , );
rtDeclareVariable(float, time0, , );
rtDeclareVariable(float, time1, , );

struct Camera {
  static __device__ optix::Ray generateRay(float s, float t, DRand48 &rnd) {
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

inline __device__ vec3f missColor(const optix::Ray &ray) {
  if(light){
    const vec3f unit_direction = normalize(ray.direction);
    const float t = 0.5f*(unit_direction.y + 1.0f);
    const vec3f c = (1.0f - t) * vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
    return c;
  }
  else
    return vec3f(0.f);
}

inline __device__ vec3f color(optix::Ray &ray, DRand48 &rnd) {
  PerRayData prd;
  prd.in.randState = &rnd;
  prd.in.time = time0 + rnd() * (time1 - time0);

  vec3f attenuation = 1.f;
  
  /* iterative version of recursion, up to depth 50 */
  for (int depth = 0; depth < 50; depth++) {
    rtTrace(world, ray, prd);
    if (prd.out.scatterEvent == rayDidntHitAnything){
      // ray got 'lost' to the environment - 'light' it with miss shader
      return attenuation * missColor(ray);
    }

    else if (prd.out.scatterEvent == rayGotCancelled)
      return attenuation * prd.out.emitted;

    else { // ray is still alive, and got properly bounced
      attenuation = prd.out.emitted + attenuation * prd.out.attenuation;
      ray = optix::make_Ray(/* origin   : */ prd.out.scattered_origin.as_float3(),
                            /* direction: */ prd.out.scattered_direction.as_float3(),
                            /* ray type : */ 0,
                            /* tmin     : */ 1e-3f,
                            /* tmax     : */ RT_DEFAULT_MAX);
    }
  }
  // recursion did not terminate - cancel it
  return vec3f(0.f);
}

/*! the actual ray generation program - note this has no formal
  function parameters, but gets its paramters throught the 'pixelID'
  and 'pixelBuffer' variables/buffers declared above */
RT_PROGRAM void renderPixel() {
  int pixel_index = pixelID.y * launchDim.x + pixelID.x;
  vec3f col(0.f, 0.f, 0.f);
  DRand48 rnd;
  rnd.init(pixel_index + run * numSamples);

  for (int s = 0; s < numSamples; s++) {
    float u = float(pixelID.x + rnd()) / float(launchDim.x);
    float v = float(pixelID.y + rnd()) / float(launchDim.y);
    optix::Ray ray = Camera::generateRay(u, v, rnd);
    col += color(ray, rnd);
  }

  // the buffer keeps its previous state unless it's initialized again
  fb[pixelID] += col.as_float3();
}

