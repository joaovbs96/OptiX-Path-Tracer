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

#define RT_USE_TEMPLATED_RTCALLABLEPROGRAM 1
#include <optix_world.h>

#include "prd.h"
#include "pdfs/pdf.h"

// the 'builtin' launch index we need to render a frame
rtDeclareVariable(uint2, pixelID,   rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim,   );

// the ray related state
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData, prd, rtPayload, );

rtBuffer<float3, 2> fb; // float3 color frame buffer
rtBuffer<unsigned int, 2> seed; // uint seed buffer

rtDeclareVariable(int, samples, , ); // number of samples
rtDeclareVariable(int, run, , ); // current sample

rtDeclareVariable(rtObject, world, , ); // scene variable

rtDeclareVariable(float3, camera_lower_left_corner, , );
rtDeclareVariable(float3, camera_horizontal, , );
rtDeclareVariable(float3, camera_vertical, , );
rtDeclareVariable(float3, camera_origin, , );
rtDeclareVariable(float3, camera_u, , );
rtDeclareVariable(float3, camera_v, , );
rtDeclareVariable(float, camera_lens_radius, , );
rtDeclareVariable(float, time0, , );
rtDeclareVariable(float, time1, , );

// PDF callable programs
rtDeclareVariable(rtCallableProgramId<float(pdf_in&)>, value, , );
rtDeclareVariable(rtCallableProgramId<float3(pdf_in&, XorShift32&)>, generate, , );
rtBuffer< rtCallableProgramId<float(pdf_in&)> > scattering_pdf;

struct Camera {
  static __device__ optix::Ray generateRay(float s, float t, XorShift32 &rnd) {
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

// Clamp color values
inline __device__ vec3f clamp(const vec3f& c) {
  vec3f temp = c;
  if(temp.x > 1.f) temp.x = 1.f;
  if(temp.y > 1.f) temp.y = 1.f;
  if(temp.z > 1.f) temp.z = 1.f;

  return temp;
}

inline __device__ vec3f color(optix::Ray &ray, XorShift32 &rnd) {
  PerRayData prd;
  prd.in.randState = &rnd;
  prd.in.time = time0 + rnd() * (time1 - time0);

  vec3f current_color = 1.f;
  
  /* iterative version of recursion, up to depth 50 */
  for (int depth = 0; depth < 50; depth++) {
    rtTrace(world, ray, prd);
    if (prd.out.scatterEvent == rayDidntHitAnything) {
      // ray got 'lost' to the environment - return attenuation set by miss shader
      return current_color * prd.out.attenuation;
    }

    // ray was absorbed
    else if (prd.out.scatterEvent == rayGotCancelled)
      return current_color * prd.out.emitted;

    // ray is still alive, and got properly bounced
    else {
      if(prd.out.is_specular) {
        current_color = prd.out.attenuation * current_color;

        ray = optix::make_Ray(/* origin   : */ prd.out.origin.as_float3(),
                              /* direction: */ prd.out.direction.as_float3(),
                              /* ray type : */ 0,
                              /* tmin     : */ 1e-3f,
                              /* tmax     : */ RT_DEFAULT_MAX);
      }
      else{
        pdf_in in(prd.out.origin, prd.out.normal);
        float3 pdf_direction = generate(in, rnd);
        float pdf_val = value(in);
        
        current_color = clamp(prd.out.emitted + (prd.out.attenuation * scattering_pdf[prd.out.type](in) * current_color) / pdf_val);

        ray = optix::make_Ray(/* origin   : */ in.origin.as_float3(),
                              /* direction: */ in.scattered_direction.as_float3(),
                              /* ray type : */ 0,
                              /* tmin     : */ 1e-3f,
                              /* tmax     : */ RT_DEFAULT_MAX);
      }
    }

    // Russian Roulette
    float p = max_component(current_color);
    if(depth > 10) {
      if(rnd() >= p)
        return current_color;
      else
        current_color *= 1/p;
    }
  }
  
  // recursion did not terminate - cancel it
  return vec3f(0.f);
}

// Remove NaN values
inline __device__ vec3f de_nan(const vec3f& c) {
  vec3f temp = c;
  if(!(temp.x == temp.x)) temp.x = 0.f;
  if(!(temp.y == temp.y)) temp.y = 0.f;
  if(!(temp.z == temp.z)) temp.z = 0.f;

  return temp;
}

/*! the actual ray generation program - note this has no formal
  function parameters, but gets its paramters throught the 'pixelID'
  and 'pixelBuffer' variables/buffers declared above */
RT_PROGRAM void renderPixel() {
  XorShift32 rnd; 
  
  if(run == 0) {
    unsigned int init_index = pixelID.y * launchDim.x + pixelID.x;
    rnd.init(init_index);
  }
  else{
    rnd.state = seed[pixelID];
  }

  // initiate the color buffer if needed
  if(run == 0)
    fb[pixelID] = make_float3(0.f, 0.f, 0.f);

  float u = float(pixelID.x + rnd()) / float(launchDim.x);
  float v = float(pixelID.y + rnd()) / float(launchDim.y);
    
  // trace ray
  optix::Ray ray = Camera::generateRay(u, v, rnd);
  
  vec3f col = de_nan(color(ray, rnd));

  fb[pixelID] += col.as_float3(); // accumulate color
  seed[pixelID] = rnd.state; // save RND state
}

