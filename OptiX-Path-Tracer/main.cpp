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

// Host include
// Host side constructors and functions. We also have OptiX 
// Programs(RT_PROGRAM) that act in the device side. Note 
// that we can use host functions as usual in these headers.
#include "host_includes/scenes.h"

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
extern "C" const char embedded_exception_program[];

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

void setExceptionProgram() {
  optix::Program exceptionProgram = g_context->createProgramFromPTXString(embedded_exception_program, "exception_program");
  g_context->setExceptionProgram(/*program ID:*/0, exceptionProgram);
}

int main(int ac, char **av) {
  // Create an OptiX context
  g_context = optix::Context::create();
  g_context->setRayTypeCount(1);
  g_context->setStackSize( 3000 );
  
  // Set main parameters
  const size_t Nx = 4480;
  const size_t Ny = 1080;
  const int samples = 1000;
  const int batchSize = 10;
  int scene = 0;

  // Create and set the camera
  Camera camera;

  // Set the ray generation and miss shader program
  setRayGenProgram();
  setMissProgram();
  setExceptionProgram();

  // Create a frame buffer
  optix::Buffer fb = createFrameBuffer(Nx, Ny);
  g_context["fb"]->set(fb);

  // TODO: initialize buffer with 0 as a safety measure

  // Create the world to render
  optix::GeometryGroup world;
  switch(scene){
    case 0: 
      world = InOneWeekend(g_context, camera, Nx, Ny);
      break;
    case 1:
      world = MovingSpheres(g_context, camera, Nx, Ny);
      break;
    default:
      printf("Error: scene unknown.\n");
      return 1;
  }
  camera.set(g_context);
  
  g_context["world"]->set(world);
  g_context["numSamples"]->setInt(samples);
  g_context["run"]->setInt(0);

  // Check OptiX scene build time
  auto t0 = std::chrono::system_clock::now();
  renderFrame(0, 0);
  auto t1 = std::chrono::system_clock::now();

  auto buildTime = std::chrono::duration<double>(t1-t0).count();
  printf("Done building optix data structures, which took %4f seconds.\n", buildTime);

  // Render scene in multiple mini-batches of a small number of samples 
  // to prevent the GPU from blocking. The buffer keeps its previous 
  // state.
  int remainder = samples % batchSize;
  int runs = (samples - remainder) / batchSize;
  g_context["numSamples"]->setInt(batchSize);

  auto t2 = std::chrono::system_clock::now();
  
  for(int i = 0; i < runs; i++){
    g_context["run"]->setInt(i);
    renderFrame(Nx, Ny);
    printf("Render Progress: %.4f%%\r", float((i * batchSize * 100.f)/samples));
  }

  // Render remaining samples, if needed
  if(remainder > 0){
    g_context["run"]->setInt(runs + 1);
    g_context["numSamples"]->setInt(remainder);
    renderFrame(Nx, Ny);
  }

  auto t3 = std::chrono::system_clock::now();
  
  auto renderTime = std::chrono::duration<double>(t3 - t2).count();
  printf("Done rendering, which took %4f seconds.\n", renderTime);
       
  // Save buffer to a PNG file
	unsigned char *arr = (unsigned char *)malloc(Nx * Ny * 3 * sizeof(unsigned char));
  const vec3f *cols = (const vec3f *)fb->map();

	for (int j = Ny - 1; j >= 0; j--)
		for (int i = 0; i < Nx; i++) {
      int col_index = Nx * j + i;
			int pixel_index = (Ny - j - 1) * 3 * Nx + 3 * i;

      // average matrix of samples
      vec3f col = cols[col_index] / float(samples);
  
      // gamma correction
      col = vec3f(sqrt(col.x), sqrt(col.y), sqrt(col.z));

      // from float to RGB [0, 255]
			arr[pixel_index + 0] = int(255.99 * clamp(col.x));
			arr[pixel_index + 1] = int(255.99 * clamp(col.y));
			arr[pixel_index + 2] = int(255.99 * clamp(col.z));
    }

  std::string output = "output/box_random_1000_2.png";
  stbi_write_png((char*)output.c_str(), Nx, Ny, 3, arr, 0);
  fb->unmap();

  system("PAUSE");

  return 0;
}

