#pragma once

#include "../prd.cuh"
#include "../sampling.cuh"
#include "../vec.hpp"

// communication between hit functions and the value programs
struct PDFRecord {
  float distance;
  float3 normal;
};

// input structs for the PDF programs
struct PDFParams {
  __device__ PDFParams(PerRayData& prd)
      : origin(prd.origin),
        normal(prd.shading_normal),
        matParams(prd.matParams) {}

  const float3 origin;
  const float3 normal;
  float3 direction;
  MaterialParameters matParams;
};
