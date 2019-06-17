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
#include "host_includes/image_save.hpp"

float renderFrame(Context &g_context, int Nx, int Ny) {
  auto t0 = std::chrono::system_clock::now();

  // Validate settings
  g_context->validate();

  // Launch ray generation program
  g_context->launch(/*program ID:*/ 0, /*launch dimensions:*/ Nx, Ny);

  auto t1 = std::chrono::system_clock::now();
  auto time = std::chrono::duration<float>(t1 - t0).count();

  return (float)time;
}

int Optix_Config(App_State &app) {
  // Set RTX global attribute(should be done before creating the context)
  if (app.RTX) {
    int RTX = true;
    RTresult res;
    res = rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(RTX),
                               &(RTX));
    if (res != RT_SUCCESS) {
      printf("Error: RTX mode is required for this application, exiting. \n");
      system("PAUSE");
      exit(0);
    } else
      printf("OptiX RTX execution mode is ON.\n");
  }

  // Create an OptiX context
  app.context->setRayTypeCount(2);  // radiance rays and shadow rays
  app.context->setMaxTraceDepth(5);

  // Set samples, ray depth and russian roulette variables
  app.context["samples"]->setInt(app.samples);
  app.context["russian"]->setInt(app.russian);
  app.context["maxDepth"]->setInt(app.depth);

  // Create and set the world
  switch (app.scene) {
    case 0:  // Peter Shirley's "In One Weekend" scene
      InOneWeekend(app);
      break;

    case 1:  // Moving Spheres test scene
      MovingSpheres(app);
      break;

    case 2:  // Cornell Box scene
      Cornell(app);
      break;

    case 3:  // Peter Shirley's "The Next Week" final scene
      Final_Next_Week(app);
      break;

    case 4:  // 3D models test scene
      Test_Scene(app);
      break;

    default:
      throw "Selected scene is unknown";
  }

  // Create an output buffer
  app.accBuffer = createFrameBuffer(app.W, app.H, app.context);
  app.context["acc_buffer"]->set(app.accBuffer);

  // Create a display buffer
  app.displayBuffer = createDisplayBuffer(app.W, app.H, app.context);
  app.context["display_buffer"]->set(app.displayBuffer);

  printf("OptiX Building Time: %.2f\n", renderFrame(app.context, 0, 0));

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
  App_State app;
  float renderTime = 0.f;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    {
      if (!app.start) {
        // Create and append program params window
        ImGui::Begin("Program Parameters");

        ImGui::InputInt("Width", &app.W, 1, 100);
        ImGui::InputInt("Height", &app.H, 1, 100);

        ImGui::InputInt("Samples Per Pixel", &app.samples, 1, 100);

        ImGui::Checkbox("RTX Mode", &app.RTX);

        ImGui::Checkbox("Russian Roulette", &app.russian);

        ImGui::InputInt("Max Depth", &app.depth, 1, 100);

        ImGui::Combo("Scene", &app.scene,
                     "Peter Shirley's In One Weekend\0Peter Shirley's The Next "
                     "Week(Moving Spheres)\0Cornell Box\0Peter Shirley's The "
                     "Next Week(Final Scene)\0Model Test Scene\0");

        if (app.scene == 4)
          ImGui::Combo("model selection", &app.model,
                       "Placeholder Model\0Lucy\0Chinese "
                       "Dragon\0Spheres\0Pie\0Sponza\0");

        ImGui::Checkbox("Show Progress", &app.showProgress);

        ImGui::Text("Save as:");
        ImGui::InputText("Filename", &app.fileName, 0, 0, 0);
        ImGui::SameLine();
        ShowHelpMarker("File extension will be added automatically.");
        ImGui::Combo("Filetype", &app.fileType, ".PNG\0.HDR\0");

        // check if render button has been pressed
        if (ImGui::Button("Render")) {
          if (app.W > 0 && app.H > 0 && app.samples > 0) {
            // Configure OptiX context & scene
            Optix_Config(app);

            // start flag
            app.start = true;
            app.currentSample = 0;

            // set proportional image size
            if ((window_width < app.W) || (window_height < app.H)) {
              if (app.W > app.H) {
                Wf = window_height * 0.9f;
                Hf = ((Wf * app.H) / app.W);
              } else {
                Hf = window_width * 0.9f;
                Wf = ((Hf * app.W) / app.H);
              }
            } else {
              Wf = app.W * 1.f;
              Hf = app.H * 1.f;
            }

            // allocate preview array
            imageData = (uchar1 *)malloc(app.W * app.H * sizeof(uchar4));
          } else {
            printf("Selected settings are invalid:\n");

            if (app.samples <= 0)
              printf("- 'samples' should be a positive integer.\n");

            if (app.W <= 0) printf("- 'width' should be a positive integer.\n");

            if (app.H <= 0)
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
        app.context["frame"]->setInt(app.currentSample);
        renderTime += renderFrame(app.context, app.W, app.H);

        // copy stream buffer content
        if (app.showProgress) {
          uchar1 *copyArr = (uchar1 *)app.displayBuffer->map();
          memcpy(imageData, copyArr, app.W * app.H * sizeof(uchar4));
          app.displayBuffer->unmap();
        }

        ImGui::Text("sample = %d / %d", app.currentSample, app.samples);
        ImGui::Text("Progress: ");
        ImGui::SameLine();
        ImGui::ProgressBar(app.currentSample / float(app.samples));

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
        app.currentSample++;
      }

      ImGui::End();
    }

    // Only show the image if its respective flag is true.
    // progressively showing the progress needs more memory and might be slower
    // overall. Turn progressive rendering off if it's too slow.
    if (app.showProgress && app.start) {
      ImGui::Begin("Render Preview");

      GLuint textureId;
      glGenTextures(1, &textureId);
      glBindTexture(GL_TEXTURE_2D, textureId);

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, app.W, app.H, 0, GL_RGBA,
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
    if (app.currentSample > 0)
      if (!app.done)
        if (app.currentSample == app.samples) {
          printf("Done rendering, output file will be saved.\n");

          // Save to file type selected in the initial setup
          if (app.fileType == 0)
            Save_PNG(app, app.accBuffer);
          else
            Save_HDR(app, app.accBuffer);

          printf("Render time: %.2fs\n", renderTime);

          app.done = true;
        }

    // if we are finished, leave the render loop
    if (app.done) break;
  }

  glfwDestroyWindow(window);
  glfwTerminate();

  system("PAUSE");

  return 0;
}