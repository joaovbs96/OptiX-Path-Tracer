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

struct BRDF_Sampler {
  std::vector<Program> sample, pdf, eval;
};

struct Light_Sampler {
  std::vector<Program> sample, pdf;
  std::vector<float3> emissions;
};

void setRayGenerationProgram(Context &g_context, BRDF_Sampler &brdf,
                             Light_Sampler &lights) {
  // set raygen program of the scene
  Program raygen =
      g_context->createProgramFromPTXString(raygen_program, "renderPixel");

  // set BRDF Sample programs buffer
  Buffer sample_buffer;
  sample_buffer =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID);
  sample_buffer->setSize(NUMBER_OF_MATERIALS);

  callableProgramId<int(int)> *sample_data;
  sample_data =
      static_cast<callableProgramId<int(int)> *>(sample_buffer->map());
  for (int i = 0; i < NUMBER_OF_MATERIALS; i++)
    sample_data[i] = callableProgramId<int(int)>(brdf.sample[i]->getId());

  sample_buffer->unmap();
  raygen["BRDF_Sample"]->setBuffer(sample_buffer);

  // set BRDF PDF programs buffer
  Buffer pdf_buffer;
  pdf_buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID);
  pdf_buffer->setSize(NUMBER_OF_MATERIALS);

  callableProgramId<int(int)> *pdf_data;
  pdf_data = static_cast<callableProgramId<int(int)> *>(pdf_buffer->map());
  for (int i = 0; i < NUMBER_OF_MATERIALS; i++)
    pdf_data[i] = callableProgramId<int(int)>(brdf.pdf[i]->getId());

  pdf_buffer->unmap();
  raygen["BRDF_PDF"]->setBuffer(pdf_buffer);

  // set BRDF evaluation programs buffer
  Buffer eval_buffer;
  eval_buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID);
  eval_buffer->setSize(NUMBER_OF_MATERIALS);

  callableProgramId<int(int)> *eval_data;
  eval_data = static_cast<callableProgramId<int(int)> *>(eval_buffer->map());
  for (int i = 0; i < NUMBER_OF_MATERIALS; i++)
    eval_data[i] = callableProgramId<int(int)>(brdf.eval[i]->getId());

  eval_buffer->unmap();
  raygen["BRDF_Evaluate"]->setBuffer(eval_buffer);

  // set light sample programs buffer
  Buffer light_sample_buffer;
  light_sample_buffer =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID);
  light_sample_buffer->setSize(lights.sample.size());

  callableProgramId<int(int)> *light_sample_data;
  light_sample_data =
      static_cast<callableProgramId<int(int)> *>(light_sample_buffer->map());
  for (int i = 0; i < lights.sample.size(); i++)
    light_sample_data[i] =
        callableProgramId<int(int)>(lights.sample[i]->getId());

  light_sample_buffer->unmap();
  raygen["Light_Sample"]->setBuffer(light_sample_buffer);

  // set light PDF programs buffer
  Buffer light_pdf_buffer;
  light_pdf_buffer =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID);
  light_pdf_buffer->setSize(lights.pdf.size());

  callableProgramId<int(int)> *light_pdf_data;
  light_pdf_data =
      static_cast<callableProgramId<int(int)> *>(light_pdf_buffer->map());
  for (int i = 0; i < lights.pdf.size(); i++)
    light_pdf_data[i] = callableProgramId<int(int)>(lights.pdf[i]->getId());

  light_pdf_buffer->unmap();
  raygen["Light_PDF"]->setBuffer(light_pdf_buffer);

  // set light emissions program buffer
  Buffer emissions_buffer =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3);
  emissions_buffer->setSize(lights.emissions.size());
  float3 *emissions_data = static_cast<float3 *>(emissions_buffer->map());
  for (int i = 0; i < lights.emissions.size(); i++)
    emissions_data[i] = lights.emissions[i];

  emissions_buffer->unmap();
  raygen["Light_Emissions"]->setBuffer(emissions_buffer);

  raygen["numLights"]->setInt(lights.emissions.size());

  g_context->setEntryPointCount(1);
  g_context->setRayGenerationProgram(/*program ID:*/ 0, raygen);
}

Buffer callableProgramBuffer(std::vector<Program> &list, Context &g_context) {
  Buffer buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID,
                                          list.size());

  callableProgramId<int(int)> *data =
      static_cast<callableProgramId<int(int)> *>(buffer->map());

  for (int i = 0; i < list.size(); i++)
    data[i] = callableProgramId<int(int)>(list[i]->getId());

  buffer->unmap();

  return buffer;
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
  else if (id == IMG) {  // rgbe image background
    missProgram =
        g_context->createProgramFromPTXString(miss_program, "img_background");

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

    // set to false is it's a cylindrical map
    missProgram["isSpherical"]->setInt(true);
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