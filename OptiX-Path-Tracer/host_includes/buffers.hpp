#ifndef BUFFERSH
#define BUFFERSH

#include "host_common.hpp"

Buffer createFrameBuffer(int Nx, int Ny, Context &g_context) {
  Buffer pixelBuffer = g_context->createBuffer(RT_BUFFER_OUTPUT);
  pixelBuffer->setFormat(RT_FORMAT_FLOAT4);
  pixelBuffer->setSize(Nx, Ny);
  return pixelBuffer;
}

Buffer createDisplayBuffer(int Nx, int Ny, Context &g_context) {
  Buffer pixelBuffer = g_context->createBuffer(RT_BUFFER_OUTPUT);
  pixelBuffer->setFormat(RT_FORMAT_UNSIGNED_BYTE4);
  pixelBuffer->setSize(Nx, Ny);
  return pixelBuffer;
}

// create Callable Program id buffer
Buffer createBuffer(std::vector<Program> &list, Context &g_context) {
  Buffer buffer =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID);
  buffer->setSize(list.size());

  callableProgramId<int(int)> *data =
      static_cast<callableProgramId<int(int)> *>(buffer->map());

  for (int i = 0; i < list.size(); i++)
    data[i] = callableProgramId<int(int)>(list[i]->getId());

  buffer->unmap();

  return buffer;
}

Buffer createBuffer(Program &program, Context &g_context) {
  std::vector<Program> buffer;
  buffer.push_back(program);

  return createBuffer(buffer, g_context);
}

// Create float buffer
Buffer createBuffer(std::vector<float> &list, Context &g_context) {
  Buffer buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT);
  buffer->setSize(list.size());

  float *data = static_cast<float *>(buffer->map());

  for (int i = 0; i < list.size(); i++) data[i] = list[i];

  buffer->unmap();

  return buffer;
}

// Create float2 buffer
Buffer createBuffer(std::vector<float2> &list, Context &g_context) {
  Buffer buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2);
  buffer->setSize(list.size());

  float2 *data = static_cast<float2 *>(buffer->map());

  for (int i = 0; i < list.size(); i++) data[i] = list[i];

  buffer->unmap();

  return buffer;
}

// Create float3 buffer
Buffer createBuffer(std::vector<float3> &list, Context &g_context) {
  Buffer buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3);
  buffer->setSize(list.size());

  float3 *data = static_cast<float3 *>(buffer->map());

  for (int i = 0; i < list.size(); i++) data[i] = list[i];

  buffer->unmap();

  return buffer;
}

// Create int buffer
Buffer createBuffer(std::vector<int> &list, Context &g_context) {
  Buffer buffer = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT);
  buffer->setSize(list.size());

  int *data = static_cast<int *>(buffer->map());

  for (int i = 0; i < list.size(); i++) data[i] = list[i];

  buffer->unmap();

  return buffer;
}

// Create uint3 buffer
Buffer createBuffer(std::vector<uint3> &list, Context &g_context) {
  Buffer buffer =
      g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3);
  buffer->setSize(list.size());

  uint3 *data = static_cast<uint3 *>(buffer->map());

  for (int i = 0; i < list.size(); i++) data[i] = list[i];

  buffer->unmap();

  return buffer;
}

#endif