#ifndef GUIH
#define GUIH

#include "host_common.hpp"
#include "scenes.hpp"

#include "../lib/imgui/imgui.h"
#include "../lib/imgui/imgui_impl_glfw.h"
#include "../lib/imgui/imgui_impl_opengl3.h"
#include "../lib/imgui/imgui_stdlib.h"

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

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && \
    !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

struct GUIState {
  GUIState()
      : w(0),
        h(0),
        pW(0),
        samples(0),
        scene(0),
        model(0),
        frequency(1),
        currentSample(0),
        progressive(false),
        done(false),
        start(false),
        HDR(false),
        fileName("out") {}
  int w, h, pW, samples, scene, currentSample, model, frequency;
  bool done, start, HDR, progressive;
  Buffer accBuffer, displayBuffer;
  std::string fileName;

  int dimensions() { return w * h; }
};

// Helper to display a little (?) mark which shows a tooltip when hovered.
static void ShowHelpMarker(const char *desc) {
  ImGui::TextDisabled("(?)");
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

int Save_SB_PNG(GUIState &state, Buffer &buffer) {
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
  state.fileName += ".png";
  const char *name = (char *)state.fileName.c_str();
  return stbi_write_png(name, state.w, state.h, 3, arr, 0);
}

int Save_SB_HDR(GUIState &state, Buffer &buffer) {
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
  state.fileName += ".hdr";
  const char *name = (char *)state.fileName.c_str();
  return stbi_write_hdr(name, state.w, state.h, 3, arr);

  return 0;
}

#endif