#ifndef GUIH
#define GUIH

// dear imgui: standalone example application for GLFW + OpenGL 3, using
// programmable pipeline If you are new to dear imgui, see examples/README.txt
// and documentation at the top of imgui.cpp. (GLFW is a cross-platform general
// purpose library for handling windows, inputs, OpenGL/Vulkan graphics context
// creation, etc.)

#include <stdio.h>
#include <mutex>
#include <string>
#include "scenes.h"

/*#include "../lib/imgui/imgui.h"
#include "../lib/imgui/imgui_impl_glfw.h"
#include "../lib/imgui/imgui_impl_opengl3.h"

// About OpenGL function loaders: modern OpenGL doesn't have a standard header
// file and requires individual function pointers to be loaded manually. Helper
// libraries are often used for this purpose! Here we are supporting a few
// common ones: gl3w, glew, glad. You may use another loader/header of your
// choice (glext, glLoadGen, etc.), or chose to manually implement your own.
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h>  // Initialize with gl3wInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>  // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>  // Initialize with gladLoadGL()
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to
// maximize ease of testing and compatibility with old VS compilers. To link
// with VS2010-era libraries, VS2015+ requires linking with
// legacy_stdio_definitions.lib, which we do using this pragma. Your own project
// should not be affected, as you are likely to link with a newer binary of GLFW
// that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && \
    !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

struct ImGuiParams {
  ImGuiParams()
      : width(0),
        height(0),
        samples(0),
        scene(0),
        model(0),
        currentSample(0),
        renderedFrame(false),
        open(true),
        done(false),
        start(false),
        fileName("out") {}
  int width, height, samples, scene, currentSample, model;
  bool renderedFrame, open, done, start, close;
  std::string fileName;
  std::mutex mtx;

  int dimensions() { return width * height; }
};*/

int Save_HDR(Buffer &buffer, std::string fileName, int Nx, int Ny,
             int samples) {
  float *arr;
  arr = (float *)malloc(Nx * Ny * 3 * sizeof(float));

  const float3 *cols = (const float3 *)buffer->map();

  for (int j = Ny - 1; j >= 0; j--)
    for (int i = 0; i < Nx; i++) {
      int col_index = Nx * j + i;
      int pixel_index = (Ny - j - 1) * 3 * Nx + 3 * i;

      // average matrix of samples
      float3 col = cols[col_index] / float(samples);

      // Apply Reinhard style tone mapping
      // Eq (3) from 'Photographic Tone Reproduction for Digital Images'
      // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.164.483&rep=rep1&type=pdf
      col = col / (make_float3(1.f) + col);

      // HDR output
      arr[pixel_index + 0] = col.x;  // R
      arr[pixel_index + 1] = col.y;  // G
      arr[pixel_index + 2] = col.z;  // B
    }

  buffer->unmap();

  fileName += std::to_string(samples) + ".hdr";
  return stbi_write_hdr((char *)fileName.c_str(), Nx, Ny, 3, arr);
}

int Save_PNG(Buffer &buffer, std::string fileName, int Nx, int Ny,
             int samples) {
  unsigned char *arr;
  arr = (unsigned char *)malloc(Nx * Ny * 3 * sizeof(unsigned char));

  const float3 *cols = (const float3 *)buffer->map();

  for (int j = Ny - 1; j >= 0; j--)
    for (int i = 0; i < Nx; i++) {
      int col_index = Nx * j + i;
      int pixel_index = (Ny - j - 1) * 3 * Nx + 3 * i;

      // average matrix of samples
      float3 col = cols[col_index] / float(samples);

      // Apply gamma correction
      col = sqrt(col);

      // from float to RGB [0, 255]
      arr[pixel_index + 0] = int(255.99 * Clamp(col.x, 0.f, 1.f));  // R
      arr[pixel_index + 1] = int(255.99 * Clamp(col.y, 0.f, 1.f));  // G
      arr[pixel_index + 2] = int(255.99 * Clamp(col.z, 0.f, 1.f));  // B
    }

  buffer->unmap();

  // output png file
  fileName += std::to_string(samples) + ".png";
  return stbi_write_png((char *)fileName.c_str(), Nx, Ny, 3, arr, 0);
}

#endif