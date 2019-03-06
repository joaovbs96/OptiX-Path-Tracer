#pragma once

#define _USE_MATH_DEFINES 1
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <random>
#include <string>

#include "../programs/vec.hpp"

// encapsulates PTX string program creation
Program createProgram(const char file[], const std::string &program,
                      Context &g_context) {
  return g_context->createProgramFromPTXString(file, program);
}

float rnd() {
  static std::mt19937 gen(0);
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

// TODO: merge scene state, light sampler and brdf sampler in a single struct

struct BRDF_Sampler {
  std::vector<Program> sample, pdf, eval;
};

struct Light_Sampler {
  std::vector<Program> sample, pdf;
  std::vector<float3> emissions;
};
