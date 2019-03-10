#ifndef BUFFERSH
#define BUFFERSH

// buffers.hpp: Define buffer creation functions

#include "host_common.hpp"

/////////////////////////////
// Output buffer functions //
/////////////////////////////

// Create a frame buffer(float4) with given dimensions
Buffer createFrameBuffer(int Nx, int Ny, Context &g_context) {
  Buffer pixelBuffer = g_context->createBuffer(RT_BUFFER_OUTPUT);
  pixelBuffer->setFormat(RT_FORMAT_FLOAT4);
  pixelBuffer->setSize(Nx, Ny);
  return pixelBuffer;
}

// Create a display buffer(uchar4) with given dimensions
Buffer createDisplayBuffer(int Nx, int Ny, Context &g_context) {
  Buffer pixelBuffer = g_context->createBuffer(RT_BUFFER_OUTPUT);
  pixelBuffer->setFormat(RT_FORMAT_UNSIGNED_BYTE4);
  pixelBuffer->setSize(Nx, Ny);
  return pixelBuffer;
}

////////////////////////////
// Input buffer functions //
////////////////////////////

// Create Callable Program id buffer
Buffer createBuffer(std::vector<Program> &list, Context &g_context) {
  Buffer buffer = g_context->createBuffer(RT_BUFFER_INPUT);
  buffer->setFormat(RT_FORMAT_PROGRAM_ID);
  buffer->setSize(list.size());

  callableProgramId<int(int)> *data =
      static_cast<callableProgramId<int(int)> *>(buffer->map());

  for (int i = 0; i < list.size(); i++)
    data[i] = callableProgramId<int(int)>(list[i]->getId());

  buffer->unmap();

  return buffer;
}

// Create Callable Program id buffer
Buffer createBuffer(Program &program, Context &g_context) {
  Buffer buffer = g_context->createBuffer(RT_BUFFER_INPUT);
  buffer->setFormat(RT_FORMAT_PROGRAM_ID);
  buffer->setSize(1);

  callableProgramId<int(int)> *data =
      static_cast<callableProgramId<int(int)> *>(buffer->map());

  data[0] = callableProgramId<int(int)>(program->getId());

  buffer->unmap();

  return buffer;
}

// Create float OptiX buffer
Buffer createBuffer(std::vector<float> &list, Context &g_context) {
  Buffer buffer = g_context->createBuffer(RT_BUFFER_INPUT);
  buffer->setFormat(RT_FORMAT_FLOAT);
  buffer->setSize(list.size());

  float *data = static_cast<float *>(buffer->map());

  for (int i = 0; i < list.size(); i++) data[i] = list[i];

  buffer->unmap();

  return buffer;
}

// Create float2 OptiX buffer
Buffer createBuffer(std::vector<float2> &list, Context &g_context) {
  Buffer buffer = g_context->createBuffer(RT_BUFFER_INPUT);
  buffer->setFormat(RT_FORMAT_FLOAT2);
  buffer->setSize(list.size());

  float2 *data = static_cast<float2 *>(buffer->map());

  for (int i = 0; i < list.size(); i++) data[i] = list[i];

  buffer->unmap();

  return buffer;
}

// Create float3 OptiX buffer
Buffer createBuffer(std::vector<float3> &list, Context &g_context) {
  Buffer buffer = g_context->createBuffer(RT_BUFFER_INPUT);
  buffer->setFormat(RT_FORMAT_FLOAT3);
  buffer->setSize(list.size());

  float3 *data = static_cast<float3 *>(buffer->map());

  for (int i = 0; i < list.size(); i++) data[i] = list[i];

  buffer->unmap();

  return buffer;
}

// Create int OptiX buffer
Buffer createBuffer(std::vector<int> &list, Context &g_context) {
  Buffer buffer = g_context->createBuffer(RT_BUFFER_INPUT);
  buffer->setFormat(RT_FORMAT_INT);
  buffer->setSize(list.size());

  int *data = static_cast<int *>(buffer->map());

  for (int i = 0; i < list.size(); i++) data[i] = list[i];

  buffer->unmap();

  return buffer;
}

// Create uint3 OptiX buffer
Buffer createBuffer(std::vector<uint3> &list, Context &g_context) {
  Buffer buffer = g_context->createBuffer(RT_BUFFER_INPUT);
  buffer->setFormat(RT_FORMAT_UNSIGNED_INT3);
  buffer->setSize(list.size());

  uint3 *data = static_cast<uint3 *>(buffer->map());

  for (int i = 0; i < list.size(); i++) data[i] = list[i];

  buffer->unmap();

  return buffer;
}

#endif