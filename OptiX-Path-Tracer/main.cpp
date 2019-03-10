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
#include "host_includes/gui.hpp"

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

int Optix_Config(GUIState &state) {
  // Set RTX global attribute(should be done before creating the context)
  const int RTX = true;
  RTresult res;
  res = rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(RTX), &RTX);
  if (res != RT_SUCCESS) {
    printf("Error: RTX mode is required for this application, exiting. \n");
    system("PAUSE");
    exit(0);
  } else
    printf("OptiX RTX execution mode is ON.\n");

  // Create an OptiX context
  g_context = Context::create();
  g_context->setRayTypeCount(2);  // radiance rays and shadow rays
  g_context->setMaxTraceDepth(5);

  // Set number of samples
  g_context["samples"]->setInt(state.samples);

  // Set pixel dimension
  g_context["pixelDim"]->setInt(state.pW);

  // Create and set the world
  switch (state.scene) {
    case 0:  // Peter Shirley's "In One Weekend" scene
      InOneWeekend(g_context, state.w, state.h);
      break;

    case 1:  // Moving Spheres test scene
      MovingSpheres(g_context, state.w, state.h);
      break;

    case 2:  // Cornell Box scene
      Cornell(g_context, state.w, state.h);
      break;

    case 3:  // Peter Shirley's "The Next Week" final scene
      Final_Next_Week(g_context, state.w, state.h);
      break;

    case 4:  // 3D models test scene
      Test_Scene(g_context, state.w, state.h, state.model);
      break;

    default:
      throw "Selected scene is unknown";
  }

  // Create an output buffer
  state.accBuffer = createFrameBuffer(state.w, state.h, g_context);
  g_context["acc_buffer"]->set(state.accBuffer);

  // Create a display buffer
  state.displayBuffer = createDisplayBuffer(state.w, state.h, g_context);
  g_context["display_buffer"]->set(state.displayBuffer);

  printf("OptiX Building Time: %.2f\n", renderFrame(0, 0));

  return 0;
}

int main(int ac, char **av) {
  ImVec4 clear_color = ImVec4(0.43f, 0.43f, 0.43f, 1.00f);

  // Setup window
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) return 1;

    // Decide GL+GLSL versions
#if __APPLE__
  // GL 3.2 + GLSL 150
  const char *glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // Required on Mac
#else
  // GL 3.0 + GLSL 130
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+
  // only glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // 3.0+ only
#endif

  const GLFWvidmode *mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

  //  int window_width = mode->width;
  //  int window_height = mode->height;
  int window_width = 1366;
  int window_height = 768;

  // Create window with graphics context
  GLFWwindow *window = glfwCreateWindow(window_width, window_height,
                                        "OptiX Path Tracer", NULL, NULL);

  if (window == NULL) throw "Failed to create OpenGL window";

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync

  // Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
  bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
  bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
  bool err = gladLoadGL() == 0;
#else
  bool err = false;  // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader
                     // is likely to requires some form of initialization.
#endif
  if (err) throw "Failed to initialize OpenGL loader";

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  // Setup Platform/Renderer bindings
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  float Hf, Wf;
  uchar1 *imageData;
  GUIState state;
  float renderTime = 0.f;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    {
      if (!state.start) {
        // Create and append program params window
        ImGui::Begin("Program Parameters");

        ImGui::InputInt("Width", &state.w, 1, 100);
        ImGui::InputInt("Height", &state.h, 1, 100);

        ImGui::InputInt("Samples Per Pixel", &state.samples, 1, 100);

        ImGui::InputInt("Subpixel Size", &state.pW, 1, 100);
        ImGui::SameLine();
        ShowHelpMarker("Greatly impacts execution time.");

        ImGui::Combo("Scene", &state.scene,
                     "Peter Shirley's In One Weekend\0Peter Shirley's The Next "
                     "Week(Moving Spheres)\0Cornell Box\0Peter Shirley's The "
                     "Next Week(Final Scene)\0Model Test Scene\0");

        if (state.scene == 4)
          ImGui::Combo(
              "model selection", &state.model,
              "Placeholder Model\0Lucy\0Chinese Dragon\0Spheres\0Sponza\0");

        ImGui::Checkbox("Show Progress", &state.showProgress);

        ImGui::Text("Save as:");
        ImGui::InputText("Filename", &state.fileName, 0, 0, 0);
        ImGui::SameLine();
        ShowHelpMarker("File extension will be added automatically.");
        ImGui::Combo("Filetype", &state.fileType, ".PNG\0.JPG\0.HDR\0");

        // check if render button has been pressed
        if (ImGui::Button("Render")) {
          if (state.w > 0 && state.h > 0 && state.samples > 0) {
            // Convifugre OptiX context & scene
            Optix_Config(state);

            // start flag
            state.start = true;
            state.currentSample = 0;

            // set proportional image size
            if ((window_width < state.w) || (window_height < state.h)) {
              if (state.w > state.h) {
                Wf = window_height * 0.9f;
                Hf = ((Wf * state.h) / state.w);
              } else {
                Hf = window_width * 0.9f;
                Wf = ((Hf * state.w) / state.h);
              }
            } else {
              Wf = state.w * 1.f;
              Hf = state.h * 1.f;
            }

            // allocate preview array
            imageData = (uchar1 *)malloc(state.w * state.h * sizeof(uchar4));
          } else {
            printf("Selected settings are invalid:\n");

            if (state.samples <= 0)
              printf("- 'samples' should be a positive integer.\n");

            if (state.w <= 0)
              printf("- 'width' should be a positive integer.\n");

            if (state.h <= 0)
              printf("- 'height' should be a positive integer.\n");

            printf("\n");
          }
        }
      }

      // rendering has started
      else {
        // Create and append program params window
        ImGui::Begin("Progress");

        // render a frame
        g_context["frame"]->setInt(state.currentSample);
        renderTime += renderFrame(state.w, state.h);

        // copy stream buffer content
        if (state.showProgress) {
          uchar1 *copyArr = (uchar1 *)state.displayBuffer->map();
          memcpy(imageData, copyArr, state.w * state.h * sizeof(uchar4));
          state.displayBuffer->unmap();
        }

        ImGui::Text("sample = %d / %d", state.currentSample, state.samples);
        ImGui::Text("Progress: ");
        ImGui::SameLine();
        ImGui::ProgressBar(state.currentSample / float(state.samples));

        // check if cancel button has been pressed
        if (ImGui::Button("Cancel")) {
          // destroy window & opengl state
          glfwDestroyWindow(window);
          glfwTerminate();

          printf(
              "Rendering has been canceled, output file will not be saved.\n");
          system("PAUSE");

          return 0;
        }

        // update number of rendered samples
        state.currentSample++;
      }

      ImGui::End();
    }

    // Only show the image if 'progressive' is true.
    // progressively showing the progress needs more memory and might be slower
    // overall. Turn progressive rendering off if it's too slow.
    if (state.showProgress && state.start) {
      ImGui::Begin("Render Preview");

      GLuint textureId;
      glGenTextures(1, &textureId);
      glBindTexture(GL_TEXTURE_2D, textureId);

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, state.w, state.h, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, imageData);

      // display progress
      ImGui::Image((void *)(intptr_t)textureId, ImVec2(Wf, Hf));
      ImGui::End();

      glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Render GUI frame
    ImGui::Render();
    int display_w, display_h;
    glfwMakeContextCurrent(window);
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwMakeContextCurrent(window);
    glfwSwapBuffers(window);

    // if render successfully finished, save file
    if (state.currentSample > 0)
      if (!state.done)
        if (state.currentSample == state.samples) {
          printf("Done rendering, output file will be saved.\n");

          // Save to file type selected in the initial setup
          if (state.fileType == 0)
            Save_PNG(state, state.accBuffer);
          else if (state.fileType == 1)
            Save_JPG(state, state.accBuffer);
          else
            Save_HDR(state, state.accBuffer);

          printf("Render time: %.2fs\n", renderTime);

          state.done = true;
        }

    // if we are finished, leave the render loop
    if (state.done) break;
  }

  glfwDestroyWindow(window);
  glfwTerminate();

  system("PAUSE");

  return 0;
}