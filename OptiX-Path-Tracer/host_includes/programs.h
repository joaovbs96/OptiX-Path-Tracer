#ifndef PROGRAMSH
#define PROGRAMSH

#include <optix.h>
#include <optixu/optixpp.h>

#include "../programs/vec.h"
#include "pdfs.h"

/*! the precompiled programs/raygen.cu code (in ptx) that our
  cmake magic will precompile (to ptx) and link to the generated
  executable (ie, we can simply declare and use this here as
  'extern'.  */
extern "C" const char embedded_miss_program[];
extern "C" const char embedded_exception_program[];
extern "C" const char embedded_raygen_program[];

void setRayGenerationProgram(optix::Context &g_context, PDF &pdf) {
  // set raygen program of the scene
  optix::Program raygen = g_context->createProgramFromPTXString(embedded_raygen_program, "renderPixel");
  
  raygen["generate"]->setProgramId(pdf.assignGenerate(g_context));
  raygen["value"]->setProgramId(pdf.assignValue(g_context));

  g_context->setEntryPointCount(1);
  g_context->setRayGenerationProgram(/*program ID:*/0, raygen);
}

void setRayGenerationProgram(optix::Context &g_context) {
  // set raygen program of the scene
  optix::Program raygen = g_context->createProgramFromPTXString(embedded_raygen_program, "renderPixel");
  g_context->setEntryPointCount(1);
  g_context->setRayGenerationProgram(/*program ID:*/0, raygen);
}

void setMissProgram(optix::Context &g_context) {
  optix::Program missProgram = g_context->createProgramFromPTXString(embedded_miss_program, "miss_program");
  g_context->setMissProgram(/*program ID:*/0, missProgram);
}

void setExceptionProgram(optix::Context &g_context) {
  optix::Program exceptionProgram = g_context->createProgramFromPTXString(embedded_exception_program, "exception_program");
  g_context->setExceptionProgram(/*program ID:*/0, exceptionProgram);
}


#endif