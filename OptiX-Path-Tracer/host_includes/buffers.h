#ifndef BUFFERSH
#define BUFFERSH

#include "../programs/vec.h"

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

#endif