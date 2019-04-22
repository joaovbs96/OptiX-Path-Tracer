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

#endif