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