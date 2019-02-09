#ifndef PROGRAMSH
#define PROGRAMSH

#include "../programs/vec.h"
#include "pdfs.h"
#include "textures.h"

#include <string>

/*! The precompiled programs code (in ptx) that our cmake script
will precompile (to ptx) and link to the generated executable */
extern "C" const char miss_program[];
extern "C" const char exception_program[];
extern "C" const char raygen_program[];

void setRayGenerationProgram(Context &g_context, PDF &pdf,
                             Buffer &material_pdfs) {
  // set raygen program of the scene
  Program raygen =
      g_context->createProgramFromPTXString(raygen_program, "renderPixel");

  raygen["generate"]->setProgramId(pdf.assignGenerate(g_context));
  raygen["value"]->setProgramId(pdf.assignValue(g_context));
  raygen["scattering_pdf"]->setBuffer(material_pdfs);

  g_context->setEntryPointCount(1);
  g_context->setRayGenerationProgram(/*program ID:*/ 0, raygen);
}

void setRayGenerationProgram(Context &g_context) {
  // set raygen program of the scene
  Program raygen =
      g_context->createProgramFromPTXString(raygen_program, "renderPixel");
  g_context->setEntryPointCount(1);
  g_context->setRayGenerationProgram(/*program ID:*/ 0, raygen);
}

typedef enum { SKY, DARK, BOX, IMG, HDR } Miss_Programs;

void setMissProgram(Context &g_context, Miss_Programs id,
                    std::string fileName = "file") {
  Program missProgram;

  if (id == SKY)  // Blue sky pattern
    missProgram = g_context->createProgramFromPTXString(miss_program, "sky");
  else if (id == DARK)  // Dark/Black background
    missProgram = g_context->createProgramFromPTXString(miss_program, "dark");
  else if (id == BOX)  // TODO: implement a proper skybox
    missProgram = g_context->createProgramFromPTXString(miss_program, "box");
  else if (id == IMG) {  // Spherical Environmental Mapping
    missProgram = g_context->createProgramFromPTXString(
        miss_program, "environmental_mapping");

    Image_Texture img(fileName);
    GeometryInstance foo;

    Buffer texture_buffers =
        g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);

    callableProgramId<int(int)> *tex_data =
        static_cast<callableProgramId<int(int)> *>(texture_buffers->map());

    Program texture = img.assignTo(g_context);
    tex_data[0] = callableProgramId<int(int)>(texture->getId());

    texture_buffers->unmap();

    missProgram["sample_texture"]->setBuffer(texture_buffers);
  } else if (id == HDR) {
    missProgram = g_context->createProgramFromPTXString(
        miss_program, "environmental_mapping");

    HDR_Texture img(fileName);
    GeometryInstance foo;

    Buffer texture_buffers =
        g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);

    callableProgramId<int(int)> *tex_data =
        static_cast<callableProgramId<int(int)> *>(texture_buffers->map());

    Program texture = img.assignTo(g_context);
    tex_data[0] = callableProgramId<int(int)>(texture->getId());

    texture_buffers->unmap();

    missProgram["sample_texture"]->setBuffer(texture_buffers);
  } else {
    printf("Error: Miss Program unknown or not yet implemented.\n");
  }

  g_context->setMissProgram(/*program ID:*/ 0, missProgram);
}

void setExceptionProgram(Context &g_context) {
  Program exceptionProgram = g_context->createProgramFromPTXString(
      exception_program, "exception_program");
  g_context->setExceptionProgram(/*program ID:*/ 0, exceptionProgram);
}

#endif