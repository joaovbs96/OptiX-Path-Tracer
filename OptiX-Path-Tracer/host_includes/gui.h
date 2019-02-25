#ifndef GUIH
#define GUIH

// dear imgui: standalone example application for GLFW + OpenGL 3, using
// programmable pipeline If you are new to dear imgui, see examples/README.txt
// and documentation at the top of imgui.cpp. (GLFW is a cross-platform general
// purpose library for handling windows, inputs, OpenGL/Vulkan graphics context
// creation, etc.)

#include <stdio.h>
#include <string>
#include "scenes.h"

#include "../lib/imgui/imgui.h"
#include "../lib/imgui/imgui_impl_glfw.h"
#include "../lib/imgui/imgui_impl_opengl3.h"
#include "../lib/imgui/imgui_stdlib.h"

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
      : w(0),
        h(0),
        samples(0),
        scene(0),
        model(0),
        frequency(0),
        currentSample(0),
        renderedFrame(false),
        progressive(false),
        open(true),
        done(false),
        start(false),
        hasStarted(false),
        HDR(false),
        fileName("out.png") {}
  int w, h, samples, scene, currentSample, model, frequency;
  bool renderedFrame, open, done, start, close, hasStarted, HDR, progressive;
  Buffer accBuffer, displayBuffer;
  std::string fileName;

  int dimensions() { return w * h; }
};

int Save_SB_PNG(ImGuiParams &state, Buffer &buffer) {
  unsigned char *arr;
  arr = (unsigned char *)malloc(state.dimensions() * 3 * sizeof(unsigned char));

  const float4 *cols = (const float4 *)buffer->map();

  for (int j = state.h - 1; j >= 0; j--)
    for (int i = 0; i < state.w; i++) {
      int index = state.w * j + i;
      int pixel_index = 3 * (state.w * j + i);

      // average & gamma correct output color
      float3 col = make_float3(cols[index].x, cols[index].y, cols[index].z);
      col = sqrt(col / float(state.samples));

      int r = int(255.99 * Clamp(col.x, 0.f, 1.f));  // R
      int g = int(255.99 * Clamp(col.y, 0.f, 1.f));  // G
      int b = int(255.99 * Clamp(col.z, 0.f, 1.f));  // B

      arr[pixel_index + 0] = r;  // R
      arr[pixel_index + 1] = g;  // G
      arr[pixel_index + 2] = b;  // B
    }

  buffer->unmap();

  // output png file
  const char *name = (char *)state.fileName.c_str();
  return stbi_write_png(name, state.w, state.h, 3, arr, 0);
}

int Save_SB_HDR(ImGuiParams &state, Buffer &buffer) {
  float *arr;
  arr = (float *)malloc(state.dimensions() * 3 * sizeof(float));

  const float4 *cols = (const float4 *)buffer->map();

  for (int j = state.h - 1; j >= 0; j--)
    for (int i = 0; i < state.w; i++) {
      int index = state.w * j + i;
      int pixel_index = 3 * (state.w * j + i);

      // average output color
      float3 col = make_float3(cols[index].x, cols[index].y, cols[index].z);
      col = col / float(state.samples);

      // Apply Reinhard style tone mapping
      // Eq (3) from 'Photographic Tone Reproduction for Digital Images'
      // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.164.483&rep=rep1&type=pdf
      col = col / (make_float3(1.f) + col);

      arr[pixel_index + 0] = col.x;  // R
      arr[pixel_index + 1] = col.y;  // G
      arr[pixel_index + 2] = col.z;  // B
    }

  buffer->unmap();

  // output hdr file
  const char *name = (char *)state.fileName.c_str();
  return stbi_write_hdr(name, state.w, state.h, 3, arr);

  return 0;
}

#endif