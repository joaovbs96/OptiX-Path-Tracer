#ifndef PROGRAMSH
#define PROGRAMSH

#include "../programs/vec.hpp"
#include "buffers.hpp"
#include "pdfs.hpp"
#include "textures.hpp"

#include <string>

/*! The precompiled programs code (in ptx) that our cmake script
will precompile (to ptx) and link to the generated executable */
extern "C" const char miss_program[];
extern "C" const char exception_program[];
extern "C" const char raygen_program[];

struct BRDF_Sampler {
  std::vector<Program> sample, pdf, eval;
};

struct Light_Sampler {
  std::vector<Program> sample, pdf;
  std::vector<float3> emissions;
};

void setRayGenerationProgram(Context &g_context, BRDF_Sampler &brdf,
                             Light_Sampler &lights) {
  // create raygen program of the scene
  Program raygen =
      g_context->createProgramFromPTXString(raygen_program, "renderPixel");

  // BRDF callable program buffers
  raygen["BRDF_Sample"]->setBuffer(createBuffer(brdf.sample, g_context));
  raygen["BRDF_PDF"]->setBuffer(createBuffer(brdf.pdf, g_context));
  raygen["BRDF_Evaluate"]->setBuffer(createBuffer(brdf.eval, g_context));

  // Light sampling params and buffers
  raygen["Light_Sample"]->setBuffer(createBuffer(lights.sample, g_context));
  raygen["Light_PDF"]->setBuffer(createBuffer(lights.pdf, g_context));
  raygen["Light_Emissions"]->setBuffer(
      createBuffer(lights.emissions, g_context));
  raygen["numLights"]->setInt((int)lights.emissions.size());

  g_context->setEntryPointCount(1);
  g_context->setRayGenerationProgram(/*program ID:*/ 0, raygen);
}

typedef enum { SKY, DARK, BOX, IMG, HDR } Miss_Programs;

void setMissProgram(Context &g_context, Miss_Programs id,
                    std::string fileName = "file", bool isSpherical = true) {
  Program missProgram;

  if (id == SKY)  // Blue sky pattern
    missProgram = g_context->createProgramFromPTXString(miss_program, "sky");

  else if (id == DARK)  // Dark/Black background
    missProgram = g_context->createProgramFromPTXString(miss_program, "dark");

  else if (id == BOX)  // TODO: implement a proper skybox
    missProgram = g_context->createProgramFromPTXString(miss_program, "box");

  else if (id == IMG) {  // rgbe image background
    missProgram =
        g_context->createProgramFromPTXString(miss_program, "img_background");

    Image_Texture img(fileName);
    Program texture = img.assignTo(g_context);
    missProgram["sample_texture"]->setBuffer(createBuffer(texture, g_context));
  }

  else if (id == HDR) {
    missProgram = g_context->createProgramFromPTXString(
        miss_program, "environmental_mapping");

    HDR_Texture img(fileName);
    Program texture = img.assignTo(g_context);
    missProgram["sample_texture"]->setBuffer(createBuffer(texture, g_context));

    // set to false if it's a cylindrical map
    missProgram["isSpherical"]->setInt(isSpherical);
  }

  else {
    printf("Error: Miss Program unknown or not yet implemented.\n");
    system("PAUSE");
    exit(0);
  }

  g_context->setMissProgram(/*program ID:*/ 0, missProgram);
}

void setExceptionProgram(Context &g_context) {
  Program exceptionProgram = g_context->createProgramFromPTXString(
      exception_program, "exception_program");
  g_context->setExceptionProgram(/*program ID:*/ 0, exceptionProgram);
}

#endif