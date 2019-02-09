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

#include "pdfs/pdf.h"
#include "prd.h"

// TODO: we have three different clamp functions, try to merge them

// the 'builtin' launch index we need to render a frame
rtDeclareVariable(uint2, pixelID, rtLaunchIndex, );
rtDeclareVariable(uint2, launchDim, rtLaunchDim, );

// the ray related state
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData, prd, rtPayload, );

rtBuffer<float3, 2> fb;          // float3 color frame buffer
rtBuffer<unsigned int, 2> seed;  // uint seed buffer

rtDeclareVariable(int, samples, , );  // number of samples
rtDeclareVariable(int, frame, , );    // current frame

rtDeclareVariable(rtObject, world, , );  // scene variable

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
rtDeclareVariable(rtCallableProgramId<float3(pdf_in&, XorShift32&)>, generate,
                  , );
rtBuffer<rtCallableProgramId<float(pdf_in&)>> scattering_pdf;

struct Camera {
  static RT_FUNCTION Ray generateRay(float s, float t, XorShift32& rnd) {
    const float3 rd = camera_lens_radius * random_in_unit_disk(rnd);
    const float3 lens_offset = camera_u * rd.x + camera_v * rd.y;
    const float3 origin = camera_origin + lens_offset;
    const float3 direction = camera_lower_left_corner + s * camera_horizontal +
                             t * camera_vertical - origin;

    return make_Ray(/* origin   : */ origin,
                    /* direction: */ direction,
                    /* ray type : */ 0,
                    /* tmin     : */ 1e-6f,
                    /* tmax     : */ RT_DEFAULT_MAX);
  }
};

// Clamp color values
RT_FUNCTION float3 clamp(const float3& c) {
  float3 temp = c;
  if (temp.x > 1.f) temp.x = 1.f;
  if (temp.y > 1.f) temp.y = 1.f;
  if (temp.z > 1.f) temp.z = 1.f;

  return temp;
}

RT_FUNCTION float3 color(Ray& ray, XorShift32& rnd) {
  PerRayData prd;
  prd.in.randState = &rnd;
  prd.in.time = time0 + rnd() * (time1 - time0);

  float3 current_color = make_float3(1.f);

  /* iterative version of recursion, up to depth 50 */
  for (int depth = 0; depth < 50; depth++) {
    rtTrace(world, ray, prd);
    if (prd.out.scatterEvent == rayDidntHitAnything) {
      // ray got 'lost' to the environment
      // return attenuation set by miss shader
      return current_color * prd.out.attenuation;
    }

    // ray was absorbed
    else if (prd.out.scatterEvent == rayGotCancelled)
      return current_color * prd.out.emitted;

    // ray is still alive, and got properly bounced
    else {
      if (prd.out.is_specular) {
        current_color = prd.out.attenuation * current_color;

        ray = make_Ray(/* origin   : */ prd.out.origin,
                       /* direction: */ prd.out.direction,
                       /* ray type : */ 0,
                       /* tmin     : */ 1e-3f,
                       /* tmax     : */ RT_DEFAULT_MAX);
      } else {
        pdf_in in(prd.out.origin, prd.out.normal);
        float3 pdf_direction = generate(in, rnd);
        float pdf_val = value(in);

        current_color =
            clamp(prd.out.emitted +
                  (prd.out.attenuation * scattering_pdf[prd.out.type](in) *
                   current_color) /
                      pdf_val);

        ray = make_Ray(/* origin   : */ in.origin,
                       /* direction: */ in.scattered_direction,
                       /* ray type : */ 0,
                       /* tmin     : */ 1e-3f,
                       /* tmax     : */ RT_DEFAULT_MAX);
      }
    }

    // Russian Roulette Path Termination
    float p = max_component(current_color);
    if (depth > 10) {
      if (rnd() >= p)
        return current_color;
      else
        current_color *= 1.f / p;
    }
  }

  // recursion did not terminate - cancel it
  return make_float3(0.f);
}

// Remove NaN values
RT_FUNCTION float3 de_nan(const float3& c) {
  float3 temp = c;
  if (!(temp.x == temp.x)) temp.x = 0.f;
  if (!(temp.y == temp.y)) temp.y = 0.f;
  if (!(temp.z == temp.z)) temp.z = 0.f;

  return temp;
}

/*! the actual ray generation program - note this has no formal
  function parameters, but gets its paramters throught the 'pixelID'
  and 'pixelBuffer' variables/buffers declared above */
RT_PROGRAM void renderPixel() {
  XorShift32 rnd;

  // init frame buffer and rng
  if (frame == 0) {
    unsigned int init_index = pixelID.y * launchDim.x + pixelID.x;
    rnd.init(init_index);

    // initiate the color buffer
    fb[pixelID] = make_float3(0.f, 0.f, 0.f);
  } else
    rnd.state = seed[pixelID];

  // Subpixel jitter: send the ray through a different position inside the pixel
  // each time, to provide antialiasing.
  float u = float(pixelID.x + rnd()) / float(launchDim.x);
  float v = float(pixelID.y + rnd()) / float(launchDim.y);

  // trace ray
  Ray ray = Camera::generateRay(u, v, rnd);

  fb[pixelID] += de_nan(color(ray, rnd));  // accumulate color
  seed[pixelID] = rnd.state;               // save RND state
}
