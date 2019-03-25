#ifndef GUIH
#define GUIH

// gui.hpp: Define functions and include libraries used by GUI

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

// Struct used to keep GUI state
struct GUIState {
  GUIState()
      : w(500),
        h(500),
        pW(1),
        samples(500),
        scene(2),
        model(0),
        frequency(1),
        currentSample(0),
        showProgress(true),
        done(false),
        start(false),
        fileType(0),
        fileName("out") {}
  int w, h, pW, samples, scene, currentSample, model, frequency, fileType;
  bool done, start, showProgress;
  Buffer accBuffer, displayBuffer;
  std::string fileName;
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

// Save OptiX output buffer to .PNG file
int Save_PNG(GUIState &state, Buffer &buffer) {
  unsigned char *arr;
  arr = (unsigned char *)malloc(state.w * state.h * 3 * sizeof(unsigned char));

  const float4 *cols = (const float4 *)buffer->map();

  for (int j = state.h - 1; j >= 0; j--)
    for (int i = 0; i < state.w; i++) {
      int index = state.w * j + i;
      int pixel_index = 3 * (state.w * j + i);

      // average & gamma correct output color
      float3 col = make_float3(cols[index].x, cols[index].y, cols[index].z);
      col = sqrt(col / float(state.samples));

      // Clamp and convert to [0, 255]
      col = 255.99f * clamp(col, 0.f, 1.f);

      // Copy int values to array
      arr[pixel_index + 0] = (int)col.x;  // R
      arr[pixel_index + 1] = (int)col.y;  // G
      arr[pixel_index + 2] = (int)col.z;  // B
    }

  buffer->unmap();

  // Save .PNG file
  state.fileName += ".png";
  const char *name = (char *)state.fileName.c_str();
  return stbi_write_png(name, state.w, state.h, 3, arr, 0);
}

// TODO: not working
// Save OptiX output buffer to .JPG file
int Save_JPG(GUIState &state, Buffer &buffer, int quality = 99) {
  if (quality < 1 || quality > 100) throw "Invalid JPG quality value";

  unsigned char *arr;
  arr = (unsigned char *)malloc(state.w * state.h * 3 * sizeof(unsigned char));

  const float4 *cols = (const float4 *)buffer->map();

  for (int j = state.h - 1; j >= 0; j--)
    for (int i = 0; i < state.w; i++) {
      int index = state.w * j + i;
      int pixel_index = 3 * (state.w * j + i);

      // average & gamma correct output color
      float3 col = make_float3(cols[index].x, cols[index].y, cols[index].z);
      col = sqrt(col / float(state.samples));

      // Clamp and convert to [0, 255]
      col = 255.99f * clamp(col, 0.f, 1.f);

      // Copy color values to array
      arr[pixel_index + 0] = (int)col.x;  // R
      arr[pixel_index + 1] = (int)col.y;  // G
      arr[pixel_index + 2] = (int)col.z;  // B
    }

  buffer->unmap();

  // Save .JPG file
  state.fileName += ".jpg";
  const char *name = (char *)state.fileName.c_str();
  return stbi_write_jpg(name, state.w, state.h, 3, arr, quality);
}

// Save OptiX output buffer to .HDR file
int Save_HDR(GUIState &state, Buffer &buffer) {
  float *arr;
  arr = (float *)malloc(state.w * state.h * 3 * sizeof(float));

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

  // Save .HDR file
  state.fileName += ".hdr";
  const char *name = (char *)state.fileName.c_str();
  return stbi_write_hdr(name, state.w, state.h, 3, arr);

  return 0;
}

#endif