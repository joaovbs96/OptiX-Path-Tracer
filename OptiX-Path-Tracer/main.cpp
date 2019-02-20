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

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

// Host side constructors and functions
#include "host_includes/scenes.h"
//#include "host_includes/scene_parser.h"

#define HDR_OUTPUT TRUE

Context g_context;

float renderFrame(int Nx, int Ny) {
  auto t0 = std::chrono::system_clock::now();

  // Validate settings
  g_context->validate();

  // Launch ray generation program
  g_context->launch(/*program ID:*/ 0, /*launch dimensions:*/ Nx, Ny);

  auto t1 = std::chrono::system_clock::now();
  auto time = std::chrono::duration<float>(t1 - t0).count();

  return (float)time;
}

Buffer createFrameBuffer(int Nx, int Ny) {
  Buffer pixelBuffer = g_context->createBuffer(RT_BUFFER_OUTPUT);
  pixelBuffer->setFormat(RT_FORMAT_FLOAT3);
  pixelBuffer->setSize(Nx, Ny);
  return pixelBuffer;
}

Buffer createSeedBuffer(int Nx, int Ny) {
  Buffer pixelBuffer = g_context->createBuffer(RT_BUFFER_OUTPUT);
  pixelBuffer->setFormat(RT_FORMAT_UNSIGNED_INT);
  pixelBuffer->setSize(Nx, Ny);
  return pixelBuffer;
}

int main(int ac, char **av) {
  // Set RTX global attribute
  // Should be done before creating the context
  const int RTX = true;
  if (rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(RTX), &RTX) !=
      RT_SUCCESS)
    printf("Error setting RTX mode. \n");
  else
    printf("OptiX RTX execution mode is %s.\n", (RTX) ? "on" : "off");

  // Create an OptiX context
  g_context = Context::create();
  g_context->setRayTypeCount(2);
  g_context->setStackSize(5000);
  // it's recommended to keep it under 10k, it's per core
  // TODO: investigate new OptiX stack size API(sets number of recursions rather
  // than bytes)

  // Main parameters
  int Nx, Ny;
  int scene = 4;

  // Set number of samples
  const int samples = 100;
  g_context["samples"]->setInt(samples);

  // Create and set the world
  std::string output;
  switch (scene) {
    case 0:  // Peter Shirley's "In One Weekend" scene
      Nx = Ny = 1080;
      output = "output/iow-";
      InOneWeekend(g_context, Nx, Ny);
      break;

    case 1:  // Moving Spheres test scene
      Nx = Ny = 1080;
      output = "output/moving-";
      MovingSpheres(g_context, Nx, Ny);
      break;

    case 2:  // Cornell Box scene
      Nx = Ny = 1080;
      output = "output/royl-";
      Cornell(g_context, Nx, Ny);
      break;

    case 3:  // Peter Shirley's "The Next Week" final scene
      Nx = Ny = 1080;
      output = "output/tnw-final-";
      Final_Next_Week(g_context, Nx, Ny);
      break;

    case 4:  // 3D models test scene
      Nx = Ny = 1080;
      output = "output/3D-models-";
      Test_Scene(g_context, Nx, Ny);
      break;

    case 5:  // Scene parser
      Nx = Ny = 1080;
      output = "output/parsed-";
      // Parser(g_context, "main.json");
      printf("Error: Scene parser not yet implemented.\n");
      system("PAUSE");
      return 1;
      break;

    default:
      printf("Error: scene unknown.\n");
      system("PAUSE");
      return 1;
  }

  // Create a frame buffer
  Buffer fb = createFrameBuffer(Nx, Ny);
  g_context["fb"]->set(fb);

  // Create a rng seed buffer
  Buffer seed = createSeedBuffer(Nx, Ny);
  g_context["seed"]->set(seed);

  // Build scene
  g_context["frame"]->setInt(0);
  float buildTime = renderFrame(0, 0);
  printf("Done building OptiX data structures, which took %.2f seconds.\n",
         buildTime);

  // Render scene
  float renderTime = 0.f;
  for (int i = 0; i < samples; i++) {
    g_context["frame"]->setInt(i);
    renderTime += renderFrame(Nx, Ny);

    printf("Progress: %.2f%%\r", (i * 100.f / samples));
  }
  printf("Done rendering, which took %.2f seconds.\n", renderTime);

#if HDR_OUTPUT == TRUE

  // Save frame buffer to a HDR file
  float *arr = (float *)malloc(Nx * Ny * 3 * sizeof(float));

#else

  // Save frame buffer to a PNG file
  unsigned char *arr =
      (unsigned char *)malloc(Nx * Ny * 3 * sizeof(unsigned char));

#endif
  const float3 *cols = (const float3 *)fb->map();

  for (int j = Ny - 1; j >= 0; j--)
    for (int i = 0; i < Nx; i++) {
      int col_index = Nx * j + i;
      int pixel_index = (Ny - j - 1) * 3 * Nx + 3 * i;

      // average matrix of samples
      float3 col = cols[col_index] / float(samples);

#if HDR_OUTPUT == TRUE
      // Apply Reinhard style tone mapping
      // Eq (3) from 'Photographic Tone Reproduction for Digital Images'
      // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.164.483&rep=rep1&type=pdf
      col = col / (make_float3(1.f) + col);

      // HDR output
      arr[pixel_index + 0] = col.x;  // R
      arr[pixel_index + 1] = col.y;  // G
      arr[pixel_index + 2] = col.z;  // B
#else
      // Apply gamma correction
      col = sqrt(col);

      // from float to RGB [0, 255]
      arr[pixel_index + 0] = int(255.99 * Clamp(col.x, 0.f, 1.f));  // R
      arr[pixel_index + 1] = int(255.99 * Clamp(col.y, 0.f, 1.f));  // G
      arr[pixel_index + 2] = int(255.99 * Clamp(col.z, 0.f, 1.f));  // B
#endif
    }

#if HDR_OUTPUT == TRUE
  // output hdr file
  output += std::to_string(samples) + ".hdr";
  stbi_write_hdr((char *)output.c_str(), Nx, Ny, 3, arr);

#else
  // output png file
  output += std::to_string(samples) + ".png";
  stbi_write_png((char *)output.c_str(), Nx, Ny, 3, arr, 0);

#endif

  fb->unmap();

  system("PAUSE");

  return 0;
}
