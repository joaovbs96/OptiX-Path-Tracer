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

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

// Host includes
// These are host side constructors and functions. We also have OptiX 
// Programs(RT_PROGRAM) that act in the device side. Note that we can
// use host functions as usual in the functions of these headers.
#include "host_includes/camera.h"
#include "host_includes/scenes.h"
#include "host_includes/materials.h"
#include "host_includes/hitables.h"

// Image I/O - disregard lib warnings
#pragma warning(push, 0)        
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"
#pragma warning(pop)

optix::Context g_context;

/*! the precompiled programs/raygen.cu code (in ptx) that our
  cmake magic will precompile (to ptx) and link to the generated
  executable (ie, we can simply declare and use this here as
  'extern'.  */
extern "C" const char embedded_raygen_program[];
extern "C" const char embedded_miss_program[];

// Clamp color values when saving to file
inline float clamp(float value) {
	return value > 1.0f ? 1.0f : value;
}

void renderFrame(int Nx, int Ny) {
  // Validate settings
  g_context->validate();

  // Launch ray generation program
  g_context->launch(/*program ID:*/0, /*launch dimensions:*/Nx, Ny);
}

optix::Buffer createFrameBuffer(int Nx, int Ny) {
  optix::Buffer pixelBuffer = g_context->createBuffer(RT_BUFFER_OUTPUT);
  pixelBuffer->setFormat(RT_FORMAT_FLOAT3);
  pixelBuffer->setSize(Nx, Ny);
  return pixelBuffer;
}

void setRayGenProgram() {
  optix::Program rayGenAndBackgroundProgram = g_context->createProgramFromPTXString(embedded_raygen_program, "renderPixel");
  g_context->setEntryPointCount(1);
  g_context->setRayGenerationProgram(/*program ID:*/0, rayGenAndBackgroundProgram);
}

void setMissProgram() {
  optix::Program missProgram = g_context->createProgramFromPTXString(embedded_miss_program, "miss_program");
  g_context->setMissProgram(/*program ID:*/0, missProgram);
}

int main(int ac, char **av) {
  // Create an OptiX context
  g_context = optix::Context::create();
  g_context->setRayTypeCount(1);
  g_context->setStackSize( 3000 );
  
  // Set main parameters
  const size_t Nx = 4480;
  const size_t Ny = 1080;
  const int samples = 128;
  int scene = 1;

  // Create and set the camera
  const vec3f lookfrom(13, 2, 3);
  const vec3f lookat(0, 0, 0);
  const vec3f up(0, 1, 0);
  const float fovy(20.0);
  const float aspect(float(Nx) / float(Ny));
  const float aperture(0.1f);
  const float dist(10.f);
  Camera camera(lookfrom, lookat, up, fovy, aspect, aperture, dist, 0.0, 1.0);
  camera.set(g_context);

  // Set the ray generation and miss shader program
  setRayGenProgram();
  setMissProgram();

  // Create a frame buffer
  optix::Buffer fb = createFrameBuffer(Nx, Ny);
  g_context["fb"]->set(fb);

  // Create the world to render
  optix::GeometryGroup world;
  switch(scene){
    case 0: 
      world = InOneWeekend(g_context);
      break;
    case 1:
      world = MovingSpheres(g_context);
      break;
    default:
      printf("Error: scene unknown.\n");
      return 1;
  }  
  
  g_context["world"]->set(world);
  g_context["numSamples"]->setInt(samples);

  // Check OptiX scene build time
  auto t0 = std::chrono::system_clock::now();
  renderFrame(0, 0);
  auto t1 = std::chrono::system_clock::now();

  auto buildTime = std::chrono::duration<double>(t1-t0).count();
  printf("Done building optix data structures, which took %4f seconds.\n", buildTime);

  // Render scene
  auto t2 = std::chrono::system_clock::now();
  renderFrame(Nx, Ny);
  auto t3 = std::chrono::system_clock::now();
  
  auto renderTime = std::chrono::duration<double>(t3-t2).count();
  printf("Done rendering, which took %4f seconds.\n", renderTime);
       
  // Save buffer to a PNG file
	unsigned char *arr = (unsigned char *)malloc(Nx * Ny * 3 * sizeof(unsigned char));
  const vec3f *cols = (const vec3f *)fb->map();

	for (int j = Ny - 1; j >= 0; j--)
		for (int i = 0; i < Nx; i++) {
      int col_index = Nx * j + i;
			int pixel_index = (Ny - j - 1) * 3 * Nx + 3 * i;

			arr[pixel_index + 0] = int(255.99 * clamp(cols[col_index].x));
			arr[pixel_index + 1] = int(255.99 * clamp(cols[col_index].y));
			arr[pixel_index + 2] = int(255.99 * clamp(cols[col_index].z));
    }

  std::string output = "output/moving.png";
  stbi_write_png((char*)output.c_str(), Nx, Ny, 3, arr, 0);
  fb->unmap();

  system("PAUSE");

  return 0;
}

