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
#include "host_includes/gui.h"

#define HDR_OUTPUT TRUE

Context g_context;

Buffer createFrameBuffer(int Nx, int Ny) {
  Buffer pixelBuffer = g_context->createBuffer(RT_BUFFER_OUTPUT);
  pixelBuffer->setFormat(RT_FORMAT_FLOAT4);
  pixelBuffer->setSize(Nx, Ny);
  return pixelBuffer;
}

Buffer createStreamBuffer(int Nx, int Ny, float gamma = 1.f) {
  Buffer pixelBuffer = g_context->createBuffer(RT_BUFFER_PROGRESSIVE_STREAM);
  pixelBuffer->setFormat(RT_FORMAT_UNSIGNED_BYTE4);

  pixelBuffer->setAttribute(RT_BUFFER_ATTRIBUTE_STREAM_GAMMA, sizeof(float),
                            (void *)&gamma);

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
  int scene = 1;
  bool HDR = false;

  // Set number of samples
  const int samples = 1000;
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
      output = "output/moving/moving-";
      MovingSpheres(g_context, Nx, Ny);
      break;

    case 2:  // Cornell Box scene
      Nx = Ny = 1080;
      output = "output/cornell-";
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
      Test_Scene(g_context, Nx, Ny, 0);
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

  // Create a stream buffer
  Buffer sb = createStreamBuffer(Nx, Ny);
  sb->bindProgressiveStream(fb);

  float renderTime = 0.f;
  auto t0 = std::chrono::system_clock::now();

  // Validate settings
  g_context->validate();

  // Launch progressive ray generation program
  g_context->launchProgressive(0, Nx, Ny, samples);

  // Render scene
  uint s = 0;
  bool ready = false;
  do {
    ready = sb->getProgressiveUpdateReady(s);
    printf("Progress: %.2f%%\r", (s * 100.f / samples));

    // save intermediate buffer state
    // if (s % 100 == 0) Save_SB_PNG(sb, output, Nx, Ny, s);
  } while (s < samples);

  auto t1 = std::chrono::system_clock::now();
  renderTime = std::chrono::duration<float>(t1 - t0).count();
  printf("Done rendering, which took %.2f seconds.\n", renderTime);

  if (HDR)
    Save_HDR(fb, output, Nx, Ny, samples);
  else
    Save_PNG(fb, output, Nx, Ny, samples);

  system("PAUSE");

  return 0;
}