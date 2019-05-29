#pragma once

#define _USE_MATH_DEFINES 1
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <random>
#include <string>

#include "../programs/vec.hpp"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

#include "../lib/HDRloader.h"

// encapsulates PTX string program creation
Program createProgram(const char file[], const std::string &name,
                      Context &g_context) {
  Program program = g_context->createProgramFromPTXString(file, name);

  if (rtProgramValidate(program->get()) != RT_SUCCESS) {
    throw "Program " + name + " is not complete.\n";
  }

  return program;
}

float rnd() {
  static std::mt19937 gen(0);
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

struct Light_Sampler {
  std::vector<Program> sample, pdf;
  std::vector<float3> emissions;
};

// returns smallest integer not less than a scalar or each vector component
float saturate(float x) { return ffmax(0.f, ffmin(1.f, x)); }

float Sign(float x) {
  if (x < 0.0f)
    return -1.0f;

  else if (x > 0.0f)
    return 1.0f;

  else
    return 0.0f;
}