#ifndef PROGRAMSH
#define PROGRAMSH

#include "../programs/vec.h"
#include "textures.h"
#include "pdfs.h"

#include <string>

/*! The precompiled programs code (in ptx) that our cmake script 
will precompile (to ptx) and link to the generated executable */
extern "C" const char embedded_miss_program[];
extern "C" const char embedded_exception_program[];
extern "C" const char embedded_raygen_program[];

void setRayGenerationProgram(optix::Context &g_context, PDF &pdf, optix::Buffer &material_pdfs) {
  // set raygen program of the scene
  optix::Program raygen = g_context->createProgramFromPTXString(embedded_raygen_program, "renderPixel");
  
  raygen["generate"]->setProgramId(pdf.assignGenerate(g_context));
  raygen["value"]->setProgramId(pdf.assignValue(g_context));
  raygen["scattering_pdf"]->setBuffer(material_pdfs);

  g_context->setEntryPointCount(1);
  g_context->setRayGenerationProgram(/*program ID:*/0, raygen);
}

void setRayGenerationProgram(optix::Context &g_context) {
  // set raygen program of the scene
  optix::Program raygen = g_context->createProgramFromPTXString(embedded_raygen_program, "renderPixel");
  g_context->setEntryPointCount(1);
  g_context->setRayGenerationProgram(/*program ID:*/0, raygen);
}

typedef enum {
  SKY,
  DARK,
  BOX,
  IMG,
  HDR
} Miss_Programs;

void setMissProgram(optix::Context &g_context, Miss_Programs id, std::string fileName="file") {
  optix::Program missProgram;

  if(id == SKY) // Blue sky pattern
    missProgram = g_context->createProgramFromPTXString(embedded_miss_program, "sky");
  else if(id == DARK) // Dark/Black background
    missProgram = g_context->createProgramFromPTXString(embedded_miss_program, "dark");
  else if(id == BOX) //TODO: implement a proper skybox
    missProgram = g_context->createProgramFromPTXString(embedded_miss_program, "box");
  else if(id == IMG) { // Spherical Environmental Mapping
    missProgram = g_context->createProgramFromPTXString(embedded_miss_program, "environmental_mapping");

    Image_Texture img(fileName);
    optix::GeometryInstance foo;

    optix::Buffer texture_buffers = 
              g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);

    optix::callableProgramId<int(int)>* tex_data = 
              static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());
      
    optix::Program texture = img.assignTo(g_context);
    tex_data[0] = optix::callableProgramId<int(int)>(texture->getId());
    
    texture_buffers->unmap();
    
    missProgram["sample_texture"]->setBuffer(texture_buffers);
  }
  else if(id == HDR) {
    missProgram = g_context->createProgramFromPTXString(embedded_miss_program, "environmental_mapping");

    HDR_Texture img(fileName);
    optix::GeometryInstance foo;

    optix::Buffer texture_buffers = 
              g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, 1);

    optix::callableProgramId<int(int)>* tex_data = 
              static_cast<optix::callableProgramId<int(int)>*>(texture_buffers->map());
      
    optix::Program texture = img.assignTo(g_context);
    tex_data[0] = optix::callableProgramId<int(int)>(texture->getId());
    
    texture_buffers->unmap();
    
    missProgram["sample_texture"]->setBuffer(texture_buffers);
  }
  else {
    printf("Error: Miss Program unknown or not yet implemented.\n");
  }
  
  g_context->setMissProgram(/*program ID:*/0, missProgram);
}

void setExceptionProgram(optix::Context &g_context) {
  optix::Program exceptionProgram = 
            g_context->createProgramFromPTXString(embedded_exception_program, "exception_program");
  g_context->setExceptionProgram(/*program ID:*/0, exceptionProgram);
}


#endif