#pragma once

#include "../sampling.cuh"
#include "../vec.hpp"

// communication between hit functions and the value programs
struct PDFRecord {
  float distance;
  float3 normal;
};

// input structs for the PDF programs
struct PDFParams {
  __device__ PDFParams(const float3 o, const float3 n) : origin(o), normal(n) {}

  const float3 origin;
  const float3 normal;
  float3 direction;
};