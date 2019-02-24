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
        open(true),
        done(false),
        start(false),
        hasStarted(false),
        HDR(false),
        fileName("out.png") {}
  int w, h, samples, scene, currentSample, model, frequency;
  bool renderedFrame, open, done, start, close, hasStarted, HDR, progressive;
  Buffer frame, output, stream;
  std::string fileName;

  int dimensions() { return w * h; }
};

int Save_SB_PNG(ImGuiParams &state, Buffer &buffer) {
  unsigned char *arr;
  arr = (unsigned char *)malloc(state.dimensions() * 3 * sizeof(unsigned char));

  const uchar4 *cols = (const uchar4 *)buffer->map();

  for (int j = state.h - 1; j >= 0; j--)
    for (int i = 0; i < state.w; i++) {
      int index = state.w * j + i;
      int pixel_index = 3 * (state.w * j + i);

      // from float to RGB [0, 255]
      arr[pixel_index + 0] = cols[index].x;  // R
      arr[pixel_index + 1] = cols[index].y;  // G
      arr[pixel_index + 2] = cols[index].z;  // B
    }

  buffer->unmap();

  // output png file
  const char *name = (char *)state.fileName.c_str();
  return stbi_write_png(name, state.w, state.h, 3, arr, 0);
}

int Save_SB_HDR(ImGuiParams &state, Buffer &buffer) {
  float *arr;
  arr = (float *)malloc(state.dimensions() * 3 * sizeof(float));

  const uchar4 *cols = (const uchar4 *)buffer->map();

  for (int j = state.h - 1; j >= 0; j--)
    for (int i = 0; i < state.w; i++) {
      int index = state.w * j + i;
      int pixel_index = 3 * (state.w * j + i);

      // Apply Reinhard style tone mapping
      // Eq (3) from 'Photographic Tone Reproduction for Digital Images'
      // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.164.483&rep=rep1&type=pdf
      uint4 col = make_uint4(cols[index].x, cols[index].y, cols[index].z,
                             cols[index].w);
      col = col / (make_uint4(255) + col);

      // HDR output
      arr[pixel_index + 0] = float(col.x);  // R
      arr[pixel_index + 1] = float(col.y);  // G
      arr[pixel_index + 2] = float(col.z);  // B
    }

  buffer->unmap();

  // output hdr file
  const char *name = (char *)state.fileName.c_str();
  return stbi_write_hdr(name, state.w, state.h, 3, arr);

  return 0;
}

#endif