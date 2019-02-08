#pragma once

#include "../sampling.h"
#include "../vec.h"

// communication between hit functions and the value programs
struct pdf_rec {
  float distance;
  float3 normal;
};

// input structs for the PDF programs
struct pdf_in {
  __device__ pdf_in(const float3 o, const float3 n) : origin(o), normal(n) {}

  const float3 origin;
  const float3 normal;
  float3 scattered_direction;
};